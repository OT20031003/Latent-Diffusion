import os
# 'scikit-image'ライブラリが必要です。インストールされていない場合は、
# ターミナルまたはコマンドプロンプトで以下のコマンドを実行してください:
# pip install scikit-image
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def ssim_crite(x, y):
    """2つの画像のSSIM（構造的類似性）を計算します。"""
    # data_rangeを追加して、画像の輝度範囲を正確に指定します。
    return ssim(x, y, channel_axis=-1, data_range=x.max() - x.min())

def calculate_snr_vs_ssim(sent_path, received_path):
    """
    送信ディレクトリと受信ディレクトリの画像を比較し、
    SNRごとの平均SSIMを計算します。
    """
    dic_sum = {}  # {snr: ssim の合計}
    dic_num = {}  # {snr: 画像枚数}

    # ディレクトリの存在チェック
    if not os.path.isdir(sent_path):
        print(f"エラー: ディレクトリが見つかりません: {sent_path}")
        return [], []
    if not os.path.isdir(received_path):
        print(f"エラー: ディレクトリが見つかりません: {received_path}")
        return [], []

    print(f"\n'{sent_path}' と '{received_path}' の比較処理を開始します...")

    for sp in os.listdir(sent_path):
        # 対応する画像ファイル形式を限定
        if not sp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        # ファイル名から画像IDを抽出
        img_id = "".join(filter(str.isdigit, sp))
        if not img_id:
            continue

        sent_image_path = os.path.join(sent_path, sp)
        
        for rp in os.listdir(received_path):
            if not rp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            # ファイル名が画像IDを含み、かつ一致する場合のみ処理
            if img_id in rp:
                # ファイル名からSNRと受信画像IDを抽出（ファイル名の形式に強く依存します）
                # 例: "image_-10dB_123.png" のような形式を想定
                try:
                    # '_'で分割してSNRとIDを取得するロジック（より安定）
                    parts = os.path.splitext(rp)[0].split('_')
                    if len(parts) < 2: continue

                    rimg_id_part = "".join(filter(str.isdigit, parts[-1]))
                    snr_str = parts[-2]

                    if rimg_id_part == img_id:
                        sentimg = np.array(Image.open(sent_image_path).convert('RGB').resize((256, 256)))
                        recimg = np.array(Image.open(os.path.join(received_path, rp)).convert('RGB').resize((256, 256)))
                        
                        ssim_val = ssim_crite(sentimg, recimg)
                        
                        dic_sum[snr_str] = dic_sum.get(snr_str, 0) + ssim_val
                        dic_num[snr_str] = dic_num.get(snr_str, 0) + 1
                except Exception:
                    # ファイル名の形式が想定と異なる場合はスキップ
                    continue

    if not dic_sum:
        print(f"警告: '{received_path}'内で一致する画像が見つかりませんでした。ファイル名の形式を確認してください。")
        return [], []

    xy = []
    for snr_key, total_ssim in dic_sum.items():
        try:
            # 'dB'などの文字が含まれていても数値部分を抽出
            snr_float = float("".join(filter(lambda c: c.isdigit() or c in '.-', snr_key)))
            count = dic_num[snr_key]
            avg_ssim = total_ssim / count
            xy.append((snr_float, avg_ssim))
            print(f"SNR: {snr_float} dB, 平均SSIM: {avg_ssim:.4f}")
        except (ValueError, ZeroDivisionError):
            print(f"警告: SNRキー '{snr_key}' を処理できませんでした。スキップします。")
            continue

    xy.sort() # SNRでソート
    
    x_vals = [item[0] for item in xy]
    y_vals = [item[1] for item in xy]
    
    return x_vals, y_vals

def main():
    """メイン関数：画像比較とグラフ描画を実行します。"""
    # --- 設定項目 ---
    path1 = "./sentimg"
    path2 = "./outputs/LDPC"
    # ★★★★★ ご自身の環境に合わせてパスを更新してください ★★★★★
    path3 = "./outputs/txt2img-samples" 

    # --- 計算実行 ---
    snr_ldpc, ssim_ldpc = calculate_snr_vs_ssim(path1, path2)
    snr_new, ssim_new = calculate_snr_vs_ssim(path1, path3)

    # --- グラフ描画 ---
    if not snr_ldpc and not snr_new:
        print("\nプロットするデータがありません。ディレクトリのパスやファイル名を確認してください。")
        return

    print("\nグラフを生成中...")
    plt.figure(figsize=(10, 6))
    
    # ラベル名は凡例に表示されます
    if snr_ldpc:
        plt.plot(snr_ldpc, ssim_ldpc, marker='o', linestyle='-', label='LDPC ')
    if snr_new:
        plt.plot(snr_new, ssim_new, marker='s', linestyle='-', label='Diffusion SemCom')

    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Average SSIM", fontsize=12)
    plt.title("SNR vs. Average SSIM Comparison", fontsize=14)
    plt.legend() # 凡例を表示
    plt.grid(False) # グリッド線を表示
    
    output_filename = "snr_vs_ssim_comparison.png"
    plt.savefig(output_filename)
    
    print(f"\nグラフが '{output_filename}' として保存されました。")

if __name__ == "__main__":
    main()