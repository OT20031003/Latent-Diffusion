import numpy as np
from PIL import Image
from pyldpc import make_ldpc, encode, decode
import os

path = "./sentimg" # 送信画像dirのpath
pathr = "./outputs/LDPC" # 受信画像dirのpath
"""
    列の数: n
    行の数: m = n * d_v / d_c
    1列あたりの1の数: d_v
    """
# --- パラメータ設定 ---
BLOCK_SIZE = 1024
CODE_RATE = 0.5 # 符号率
d_v, d_c = 2, 4 # LDPCのパラメータ


n = int(BLOCK_SIZE / CODE_RATE) # 符号長
k = BLOCK_SIZE # 情報長

# systematic（組織符号）の場合、Gの形状は (k, n)
try:
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    print(f"LDPC matrix created for block size {k} -> code length {n}")
except Exception as e:
    print(f"Error creating LDPC matrix: {e}")
    exit()

k_actual = G.shape[1]        # G の行数を"情報長"として使う
n_actual = G.shape[0]
print(f"Actual k = {k_actual}, n = {n_actual}")

for d in os.listdir(path):
    image = Image.open(os.path.join(path, d)).resize((256, 256))
    image_id = ""
    for i in range(len(d)):
        if d[i].isdigit():
            image_id += d[i]
    print("Processing image ID:", image_id)
    image_data = np.array(image)
    original_shape = image_data.shape
    #print(image_data.shape)  # (256, 256)
    binary_image = np.unpackbits(image_data)
    original_bits_len = len(binary_image)
    rem = len(binary_image) % k_actual
    if rem != 0:
        padded_binary_image = np.hstack((binary_image, np.zeros(k_actual - rem, dtype=np.uint8)))
    else:
        padded_binary_image = binary_image
    
    blocks = padded_binary_image.reshape(-1, k_actual)
    print(f"Image {d} divided into {blocks.shape[0]} blocks of size {k} bits.")
    
    
    for snr in range(0, 20, 1):
        decoded_blocks = []
        print(f"SNR = {snr} dB")
        for i, block in enumerate(blocks):
            print(f"Encoding block {i+1}/{blocks.shape[0]}")
            encoded_block = encode(G, block, snr)
            decoded_block = decode(H, encoded_block, snr, maxiter=100)
            
            # (Remember to apply the fix from last time here!)
            message_block = decoded_block[:k_actual] 
            decoded_blocks.append(message_block)
        decoded_binary_image = np.hstack(decoded_blocks)
        decoded_binary_image = decoded_binary_image[:original_bits_len]
        reconstructed_image_data = np.packbits(decoded_binary_image).reshape(original_shape)
        reconstructed_image = Image.fromarray(reconstructed_image_data.astype(np.uint8))
        reconstructed_image.save(os.path.join(pathr, f"output_{snr}_{image_id}.png"))
        
            