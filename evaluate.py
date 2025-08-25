import torch
import glob
import pandas as pd
from PIL import Image
from torchvision import transforms
import lpips
from torchmetrics import PeakSignalNoiseRatio
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    """
    生成された画像（サンプリングあり/なし）と元の画像を比較し、
    PSNRとLPIPSを計算して結果を表示・保存・プロットします。
    """
    # --- 設定 ---
    original_image_path = "outputs/uinput.png"
    generated_images_dir = "outputs/"
    generated_images_pattern = os.path.join(generated_images_dir, "output_*.png")
    nosample_images_pattern = os.path.join(generated_images_dir, "nosample_*.png")
    
    # --- 初期化 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    psnr = PeakSignalNoiseRatio().to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    # --- 画像の前処理 ---
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # 画像サイズを統一
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # --- 元画像の読み込み ---
    try:
        original_img = Image.open(original_image_path).convert("RGB")
        original_tensor = transform(original_img).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"エラー: 元画像 '{original_image_path}' が見つかりません。")
        print("img2img.py を実行して画像を生成したか確認してください。")
        return

    # --- 評価画像のパスを取得 ---
    generated_image_paths = sorted(glob.glob(generated_images_pattern))
    nosample_image_paths = sorted(glob.glob(nosample_images_pattern))
    
    if not generated_image_paths:
        print(f"エラー: パターン '{generated_images_pattern}' に一致する生成画像が見つかりません。")
        return

    # --- 評価の実行 ---
    results = []
    nosample_results = []
    x_ax = []
    print("--- 評価開始: サンプリングあり画像 (output_*.png) ---")
    for gen_path in generated_image_paths:
        try:
            # ▼▼▼ 変更点 ▼▼▼
            # ファイル名からSNR値を抽出する際、int() から float() に変更
            filename = os.path.basename(gen_path)
            snr_str = os.path.splitext(filename.split('_')[-1])[0]
            snr = float(snr_str)
            # ▲▲▲ 変更点 ▲▲▲
            x_ax.append(snr)
            gen_img = Image.open(gen_path).convert("RGB")
            gen_tensor = transform(gen_img).unsqueeze(0).to(device)

            psnr_score = psnr(gen_tensor, original_tensor).item()
            lpips_score = lpips_fn(original_tensor, gen_tensor).item()
            
            results.append({"SNR": snr, "PSNR": psnr_score, "LPIPS": lpips_score})
            print(f"SNR: {snr:5.1f} -> PSNR: {psnr_score:.4f}, LPIPS: {lpips_score:.4f}")
        except Exception as e:
            print(f"ファイル '{gen_path}' の処理中にエラー: {e}")

    print("\n---  (nosample_*.png) ---")
    
    for gen_path in nosample_image_paths:
        try:
            # ▼▼▼ 変更点 ▼▼▼
            # ファイル名からSNR値を抽出する際、int() から float() に変更
            filename = os.path.basename(gen_path)
            snr_str = os.path.splitext(filename.split('_')[-1])[0]
            snr = float(snr_str)
            # ▲▲▲ 変更点 ▲▲▲

            gen_img = Image.open(gen_path).convert("RGB")
            gen_tensor = transform(gen_img).unsqueeze(0).to(device)

            psnr_score = psnr(gen_tensor, original_tensor).item()
            lpips_score = lpips_fn(original_tensor, gen_tensor).item()
            
            nosample_results.append({"SNR": snr, "PSNR": psnr_score, "LPIPS": lpips_score})
            print(f"SNR: {snr:5.1f} -> PSNR: {psnr_score:.4f}, LPIPS: {lpips_score:.4f}")
        except Exception as e:
            print(f"ファイル '{gen_path}' の処理中にエラー: {e}")

    # --- 結果の表示と保存 ---
    print(x_ax)
    if results and nosample_results:
        df_sampled = pd.DataFrame(results).sort_values(by="SNR").set_index("SNR")
        df_nosample = pd.DataFrame(nosample_results).sort_values(by="SNR").set_index("SNR")
        
        
        
        # --- CSVへの保存 ---
        df_sampled.to_csv("./evaluation_results_sampled.csv")
        df_nosample.to_csv("./evaluation_results_nosample.csv")
        print(f"\n評価結果をCSVファイルに保存しました。")
        
        # --- グラフの描画 ---
        plt.figure(figsize=(10, 6))
        plt.plot(df_nosample.index, df_nosample['PSNR'],marker='x', label='No Sample')
        plt.plot(df_sampled.index, df_sampled['PSNR'],marker='o', label='Sampled')
        plt.title('PSNR Comparison')
        plt.xlabel('SNR')
        plt.ylabel('PSNR')
        plt.legend()
        plt.grid(True)
        plt.savefig('psnr_comparison.png')
        plt.close()

        # LPIPSの比較グラフを作成
        plt.figure(figsize=(10, 6))
        plt.plot(df_nosample.index, df_nosample['LPIPS'],marker='x', label='No Sample')
        plt.plot(df_sampled.index, df_sampled['LPIPS'],marker='o', label='Sampled')
        plt.title('LPIPS Comparison')
        plt.xlabel('SNR')
        plt.ylabel('LPIPS')
        plt.legend()
        plt.grid(True)
        plt.savefig('lpips_comparison.png')
        plt.close()

if __name__ == '__main__':
    main()