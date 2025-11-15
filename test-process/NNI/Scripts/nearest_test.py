import os
from PIL import Image
from time import time

# Path to the data folder next to this script
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

print("Looking in:", DATA_DIR)

images = []

# Load images
for filename in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, filename)
    try:
        img = Image.open(path)
        img.load()
        images.append(img)
        print("Loaded:", filename)
    except Exception as e:
        print("Failed to load:", filename, "Error:", e)

# If no images loaded, stop here
if not images:
    print("\n No images were loaded. Check the folder path!")
    exit()

# Test nearest interpolation
start = time()
for img in images:
    img.resize((32, 32), Image.NEAREST)
print("\nTime (NEAREST):", time() - start, "seconds")

# Save an example output
images[0].resize((32, 32), Image.NEAREST).save("nearest_output.png")
print("Saved nearest_output.png")


