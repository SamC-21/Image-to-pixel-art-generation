import os
import cv2
import shutil  
import numpy as np
from PIL import Image


DATASETS = {
    "1": {"name": "Anime Faces", "path": r"C:\Users\Sam\DatasetsDiss\DatasetFaces\images"},
    "2": {"name": "Landscapes", "path": r"C:\Users\Sam\DatasetsDiss\DatasetLandscapes"}
}

OUTPUT_DIR = "Processed_Batches"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- FUNCTIONS ---
def resize_image(img, method_choice):
    methods = {
        "1": ("Nearest", cv2.INTER_NEAREST),
        "2": ("Bilinear", cv2.INTER_LINEAR),
        "3": ("Bicubic", cv2.INTER_CUBIC)
    }
    name, flag = methods.get(method_choice, ("Bilinear", cv2.INTER_LINEAR))
    # resize line
    return cv2.resize(img, (128, 128), interpolation=flag), name

def quantize_image(img_bgr, method_choice, k):
    method_name = "unknown"
    if method_choice == "1": # K-Means
        data = img_bgr.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        output = centers[labels.flatten()].reshape(img_bgr.shape).astype(np.uint8)
        method_name = "kmeans"
    elif method_choice in ["2", "3"]: # Pillow Methods
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_method = 0 if method_choice == "2" else 2
        quantized_pil = pil_img.quantize(colors=k, method=pil_method).convert("RGB")
        output = cv2.cvtColor(np.array(quantized_pil), cv2.COLOR_RGB2BGR)
        method_name = "median_cut" if method_choice == "2" else "octree"
    return output, method_name

# --- MAIN ---
def main():

    # --- CLEANUP SECTION ---
    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing old files from {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR) # Deletes the folder and everything inside
    os.makedirs(OUTPUT_DIR)     # Creates a fresh, empty folder

    # SETUP
    print("\n--- Select Dataset ---")
    for key, info in DATASETS.items():
        print(f"[{key}] {info['name']}")
    ds_choice = input("Select dataset number: ")
    if ds_choice not in DATASETS: exit("Invalid selection.")
        
    src_path = DATASETS[ds_choice]['path']
    # Filter only images
    files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Found {len(files)} images.")

    #SETTINGS
    print("\n--- Interpolation ---")
    print("[1] Nearest  [2] Bilinear  [3] Bicubic")
    interp_choice = input("Choice: ")

    print("\n--- Quantization ---")
    print("[1] K-Means++  [2] Median Cut  [3] Octree")
    quant_choice = input("Choice: ")
    k_val = int(input("Enter K (colors): "))

    # BATCH LOOP 
    batch_size = 50
    
    # loops 50 times
    for i in range(0, len(files), batch_size):
        batch_files = files[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"\n--- Processing Batch {batch_num} (Images {i+1} to {i+len(batch_files)}) ---")

        # Process the current 50 images
        for filename in batch_files:
            full_path = os.path.join(src_path, filename)
            img = cv2.imread(full_path)
            if img is None: continue

            # Apply Logic
            resized, i_name = resize_image(img, interp_choice)
            final, q_name = quantize_image(resized, quant_choice, k_val)

            # Save
            save_name = f"{os.path.splitext(filename)[0]}_{i_name}_{q_name}_k{k_val}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), final)
        
        print(f"Batch {batch_num} saved to '{OUTPUT_DIR}' folder.")

        #checks for user to do more than 50 images at a time
        # If there are still files left, ask the user what to do
        if i + batch_size < len(files):
            user_check = input("\n>>> Batch complete. Press Enter to process next 50 (or type 'q' to stop): ")
            if user_check.lower() == 'q':
                print("Stopping process.")
                break
        else:
            print("\nAll files processed.")

if __name__ == "__main__":
    main()