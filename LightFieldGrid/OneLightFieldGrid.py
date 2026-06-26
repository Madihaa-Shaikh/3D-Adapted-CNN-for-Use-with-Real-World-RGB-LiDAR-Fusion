import cv2
import numpy as np
import os
from math import ceil, sqrt

# Path to your frames folder
folder_path = "./frames"

# Load and sort all images
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg"))])

if not image_files:
    raise FileNotFoundError("No images found in frames folder!")

images = [cv2.imread(os.path.join(folder_path, f)) for f in image_files]

# Resize images to the same size
h, w = images[0].shape[:2]
images = [cv2.resize(img, (w, h)) for img in images]

# Determine grid layout automatically
num_images = len(images)
cols = int(ceil(sqrt(num_images)))
rows = int(ceil(num_images / cols))
print(f"Arranging {num_images} images in grid {rows}x{cols}")

# Fill missing slots with black images
blank = np.zeros_like(images[0])
while len(images) < rows * cols:
    images.append(blank)

# Stack horizontally and vertically
rows_combined = [np.hstack(images[i*cols:(i+1)*cols]) for i in range(rows)]
grid = np.vstack(rows_combined)

# Save and show result
os.makedirs("results", exist_ok=True)
output_path = os.path.join("results", "lightfield_grid.png")
cv2.imwrite(output_path, grid)
print(f"Light field grid saved at: {output_path}")

cv2.imshow("Light Field Grid", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
