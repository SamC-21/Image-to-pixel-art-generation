import os
import cv2 
from PIL import Image




DATASETS = {
    "1": {
        "name": "Anime Faces",
        "path": r"C:\Users\Sam\DatasetsDiss\DatasetFaces\images"
    },
    "2": {
        "name": "Landscapes",
        "path": r"C:\Users\Sam\DatasetsDiss\DatasetLandscapes" # Change this to your second path
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
