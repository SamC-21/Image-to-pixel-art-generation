import os
import cv2 
import numpy as np
from PIL import Image

DATASETS = {
    "1": {
        "name": "Anime Faces",
        "path": r"C:\Users\Sam\DatasetsDiss\DatasetFaces\images"
    },
    "2": {
        "name": "Landscapes",
        "path": r"C:\Users\Sam\DatasetsDiss\DatasetLandscapes" 
    }
}

#choice for which dataset to use 
print("Select Dataset ---")
for key, info in DATASETS.items():
    print(f"[{key}] {info['name']} ({info['path']})")

ds_choice = input("\nSelect dataset number: ")

if ds_choice not in DATASETS:
    print("Invalid selection.")
    exit()

selected_path = DATASETS[ds_choice]['path']
# valid list of image file extensions
valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')

files = [f for f in os.listdir(selected_path)]

print(f"\n--- {DATASETS[ds_choice]['name']} contains {len(files)} images ---")
#any invalid option causes trhe program to exit
img_choice = input("\nEnter index number to process (or 'q' to quit): ")
#selects the file based on user input from the list

if img_choice.isdigit() and int(img_choice) < len(files):
    full_path = os.path.join(selected_path, files[int(img_choice)])
    img_bgr = cv2.imread(full_path)
    
    if img_bgr is None:
        exit("Could not read image.")

    # Method Selection 
    print("\n--- Quantization Methods ---")
    print("[1] K-Means++ (OpenCV)")
    print("[2] Median Cut (Pillow)")
    print("[3] Octree (Pillow)")
    
    method = input("Select method: ")
    k = int(input("Enter number of colors (e.g. 8, 16, 64): "))

    # Processing  
    if method == "1":
        # K-Means++ 
        # Reshape image data for k-means
        data = img_bgr.reshape((-1, 3)).astype(np.float32)
        # define criteria of when to stop k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # picks first k centroids using kmeans++
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        output = centers[labels.flatten()].reshape(img_bgr.shape).astype(np.uint8)
        method_name = "kmeans"

    elif method in ["2", "3"]:
        # Convert BGR to RGB for Pillow as OpenCV uses BGR by default
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 0 = Median Cut, 2 = Octree , 1 would be Max Coverage (not used here)
        pil_method = 0 if method == "2" else 2
        quantized_pil = pil_img.quantize(colors=k, method=pil_method).convert("RGB")
        
        # Convert back to BGR for saving with OpenCV
        output = cv2.cvtColor(np.array(quantized_pil), cv2.COLOR_RGB2BGR)
        method_name = "median_cut" if method == "2" else "octree"

    else:
        exit("Invalid method.")

    # seve output
    save_path = f"output_{method_name}_k{k}.png"
    cv2.imwrite(save_path, output)
    print(f"\nSaved result as {save_path}")
