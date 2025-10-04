import os
import argparse
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def compute_metric(x, y, metric='ssim'):
    """
    画像ペア x, y の類似度／誤差を計算する。
    metric: 'ssim' または 'mse'
    """
    if metric == 'ssim':
        # RGB画像を想定。data_rangeを指定して安定化。
        return ssim(x, y, channel_axis=-1, data_range=float(x.max() - x.min()))
    elif metric == 'mse':
        # MSE（平均二乗誤差）
        xd = x.astype(np.float64)
        yd = y.astype(np.float64)
        return float(np.mean((xd - yd) ** 2))
    else:
        raise ValueError("metric は 'ssim' または 'mse' を指定してください。")

def calculate_snr_vs_metric(sent_path, received_path, metric='ssim', resize=(256,256)):
    """
    送信ディレクトリと受信ディレクトリの画像を比較し、
    SNRごとの平均 metric 値を返す。
    metric: 'ssim' or 'mse'
    resize: 比較時にリサイズするサイズ (None でリサイズなし)
    """
    dic_sum = {}
    dic_num = {}

    if not os.path.isdir(sent_path):
        print(f"エラー: ディレクトリが見つかりません: {sent_path}")
        return [], []
    if not os.path.isdir(received_path):
        print(f"エラー: ディレクトリが見つかりません: {received_path}")
        return [], []

    print(f"\n'{sent_path}' と '{received_path}' の比較処理を開始します... (metric={metric})")

    for sp in os.listdir(sent_path):
        if not sp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        img_id = "".join(filter(str.isdigit, sp))
        if not img_id:
            continue

        sent_image_path = os.path.join(sent_path, sp)

        for rp in os.listdir(received_path):
            if not rp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            if img_id in rp:
                try:
                    parts = os.path.splitext(rp)[0].split('_')
                    if len(parts) < 2:
                        continue

                    rimg_id_part = "".join(filter(str.isdigit, parts[-1]))
                    snr_str = parts[-2]

                    if rimg_id_part == img_id:
                        sentimg = Image.open(sent_image_path).convert('RGB')
                        recimg = Image.open(os.path.join(received_path, rp)).convert('RGB')

                        if resize is not None:
                            sentimg = sentimg.resize(resize)
                            recimg = recimg.resize(resize)

                        sentarr = np.array(sentimg)
                        recarr = np.array(recimg)

                        # metric 計算（例外処理で堅牢に）
                        try:
                            val = compute_metric(sentarr, recarr, metric=metric)
                        except Exception as e:
                            print(f"警告: メトリック計算でエラー ({rp}): {e}")
                            continue

                        dic_sum[snr_str] = dic_sum.get(snr_str, 0.0) + val
                        dic_num[snr_str] = dic_num.get(snr_str, 0) + 1
                except Exception:
                    continue

    if not dic_sum:
        print(f"警告: '{received_path}'内で一致する画像が見つかりませんでした。ファイル名の形式を確認してください。")
        return [], []

    xy = []
    for snr_key, total in dic_sum.items():
        try:
            snr_float = float("".join(filter(lambda c: c.isdigit() or c in '.-', snr_key)))
            count = dic_num[snr_key]
            avg = total / count
            xy.append((snr_float, avg))
            print(f"SNR: {snr_float} dB, 平均{metric.upper()}: {avg:.6f} (count={count})")
        except (ValueError, ZeroDivisionError):
            print(f"警告: SNRキー '{snr_key}' を処理できませんでした。スキップします。")
            continue

    xy.sort()  # SNRでソート
    x_vals = [item[0] for item in xy]
    y_vals = [item[1] for item in xy]
    return x_vals, y_vals

def plot_results(results, title_suffix="", output_filename="snr_vs_metric.png"):
    """
    results: list of tuples (x_vals, y_vals, label)
    """
    plt.figure(figsize=(10,6))
    for x_vals, y_vals, label in results:
        if not x_vals:
            continue
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', label=label)
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Metric value", fontsize=12)
    plt.title(f"SNR vs. Metric Comparison {title_suffix}", fontsize=14)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"\nグラフが '{output_filename}' として保存されました。")

def main():
    parser = argparse.ArgumentParser(description="SNR vs SSIM/MSE 比較スクリプト")
    parser.add_argument("--sent", "-s", default="./sentimg", help="送信画像ディレクトリ")
    parser.add_argument("--recv", "-r", default="./outputs/LDPC", help="受信画像ディレクトリ (比較対象1)")
    parser.add_argument("--recv2", "-r2", default="./outputs/txt2img-samples", help="受信画像ディレクトリ (比較対象2, オプション)")
    parser.add_argument("--metric", "-m", choices=["ssim","mse","both"], default="ssim", help="使用する指標")
    parser.add_argument("--resize", type=int, nargs=2, metavar=('W','H'), default=(256,256), help="比較時のリサイズ (幅 高さ)")
    args = parser.parse_args()

    # 比較1 (LDPC)
    results = []
    if args.metric in ("ssim", "both"):
        x1, y1 = calculate_snr_vs_metric(args.sent, args.recv, metric="ssim", resize=tuple(args.resize))
        results.append((x1, y1, "LDPC - SSIM"))
        x2, y2 = calculate_snr_vs_metric(args.sent, args.recv2, metric="ssim", resize=tuple(args.resize))
        results.append((x2, y2, "DiffusionSemCom - SSIM"))

    if args.metric in ("mse", "both"):
        xm1, ym1 = calculate_snr_vs_metric(args.sent, args.recv, metric="mse", resize=tuple(args.resize))
        results.append((xm1, ym1, "LDPC - MSE"))
        xm2, ym2 = calculate_snr_vs_metric(args.sent, args.recv2, metric="mse", resize=tuple(args.resize))
        results.append((xm2, ym2, "DiffusionSemCom - MSE"))

    if not results:
        print("プロットするデータがありません。")
        return

    outname = f"snr_vs_{args.metric}.png"
    plot_results(results, title_suffix=f"({args.metric})", output_filename=outname)

if __name__ == "__main__":
    main()
