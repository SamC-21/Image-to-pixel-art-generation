import os
import cv2
import numpy as np

DATASETS = {
    "1": {"name": "Anime Faces", "path": r"C:\Users\Sam\DatasetsDiss\DatasetFaces\images"},
    "2": {"name": "Landscapes", "path": r"C:\Users\Sam\DatasetsDiss\DatasetLandscapes"}
}





# Dataset Selection
print("Select Dataset ---")
for key, info in DATASETS.items():
    print(f"[{key}] {info['name']}")

ds_choice = input("\nSelect dataset number: ")
if ds_choice not in DATASETS:
    exit("Invalid selection.")

selected_path = DATASETS[ds_choice]['path']
files = [f for f in os.listdir(selected_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Image Selection
print(f"\n--- {DATASETS[ds_choice]['name']} contains {len(files)} images ---")
img_choice = input("Enter index number to process (or 'q' to quit): ")

if img_choice.isdigit() and int(img_choice) < len(files):
    full_path = os.path.join(selected_path, files[int(img_choice)])
    img_bgr = cv2.imread(full_path)
    if img_bgr is None: exit("Could not read image.")
    
    # Converts to grayscale for edge detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Method Selection 
    print("\n--- Edge & Filter Methods ---")
    print("[1] Bilateral Filter (Denoising)")
    print("[2] Sobel Edge Detection")
    print("[3] Canny Edge Detection")
    
    method = input("Select method: ")

    # processing
    if method == "1":
        #diametere value sets the size of the pixel neighborhood
        d = int(input("Enter diameter (e.g., 9): "))
        # sigma values control the filter strength larger means more blurring
        sig = int(input("Enter sigma (e.g., 75): "))
        output = cv2.bilateralFilter(img_bgr, d, sig, sig)
        method_name = "bilateral"

    elif method == "2":
        # Pre-process with light bilateral filter to improve results
        blur = cv2.bilateralFilter(gray, 5, 50, 50)
        # Calculate gradients left to right and top to bottom
        # kernal size of 3 larger makes blurred edges more pronounced
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        # Calculate magnitude and convert back to 8-bit
        mag = cv2.magnitude(sobelx, sobely)
        output = np.uint8(np.absolute(mag))
        method_name = "sobel"

    elif method == "3":
        #lower threshold for edge linking, upper for initial edge detection
        low = int(input("Enter lower threshold (e.g., 100): "))
        high = int(input("Enter upper threshold (e.g., 200): "))
        # Pre-process with light bilateral filter
        blur = cv2.bilateralFilter(gray, 5, 50, 50)
        output = cv2.Canny(blur, low, high)
        method_name = "canny"

    else:
        exit("Invalid method.")




    # Save output
    save_path = f"test_{method_name}_output.png"
    cv2.imwrite(save_path, output)
    print(f"\nSaved result as {save_path}")