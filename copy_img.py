import os
import glob
import random
import shutil
from tqdm import tqdm

# --- Configuration ---
# 1. Specify the path to the extracted `leftImg8bit` folder from the Cityscapes dataset
source_dir = './leftImg8bit_trainvaltest/leftImg8bit' 

# 2. Destination folder name (default input path for img2img.py)
dest_dir = 'input_img'
shutil.rmtree(dest_dir)
# 3. Number of images to copy
num_files_to_copy = 10
# --- End of Configuration ---

# Create the destination folder if it does not exist
os.makedirs(dest_dir, exist_ok=True)

# Recursively search for all PNG file paths within the source_dir
# `**/*.png` means it will find all files, including those deep in subfolders
print(f"Searching for image files in '{source_dir}'...")
all_files = glob.glob(os.path.join(source_dir, '**', '*.png'), recursive=True)

if len(all_files) < num_files_to_copy:
    print(f"Error: The total number of image files ({len(all_files)}) is less than the requested number ({num_files_to_copy}).")
else:
    # Randomly select 100 files from the list
    selected_files = random.sample(all_files, num_files_to_copy)
    
    print(f"Copying {num_files_to_copy} images out of {len(all_files)} to '{dest_dir}'...")
    
    # Copy each selected file
    for file_path in tqdm(selected_files, desc="Copying images"):
        shutil.copy(file_path, dest_dir)
        
    print("Copying complete.")