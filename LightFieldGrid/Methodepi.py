import cv2
import numpy as np
import os

# 🔹 Path to your original multi-view frames
folder_path = "./frames"   # change if your frames folder is elsewhere

# 🔹 Collect and sort all frame files
image_files = sorted([f for f in os.listdir(folder_path)
                      if f.lower().endswith((".jpg", ".png", ".jpeg"))])

if not image_files:
    raise FileNotFoundError(f"❌ No image files found in {folder_path}")

# 🔹 Load all frames into memory
images = []
for f in image_files:
    path = os.path.join(folder_path, f)
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Could not read: {f}")
        continue
    # rotate only if your frames appear sideways
    # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    images.append(img)

if not images:
    raise RuntimeError("❌ No valid images could be loaded.")

print(f"✅ Loaded {len(images)} frames for EPI generation")

# All images must be same size
H, W, C = images[0].shape
images = [cv2.resize(img, (W, H)) for img in images]

# Convert to 4-D light-field array: (views, height, width, channels)
lf = np.stack(images, axis=0)

# 🔹 Function to generate EPI
def generate_epi(lightfield, slice_index=200, axis='horizontal'):
    """
    Extract a horizontal or vertical EPI from a 4-D light field.
    """
    if axis == 'horizontal':
        # take same row from each view
        epi = np.array([view[slice_index, :, :] for view in lightfield])
    else:
        # take same column from each view
        epi = np.array([view[:, slice_index, :] for view in lightfield])
    return epi

# pick central row/column to start with
row_index = H // 2
col_index = W // 2

horizontal_epi = generate_epi(lf, slice_index=row_index, axis='horizontal')
vertical_epi   = generate_epi(lf, slice_index=col_index, axis='vertical')

# 🔹 Create output folder and save
os.makedirs("results/epi", exist_ok=True)
cv2.imwrite("results/epi/horizontal_epi.png", horizontal_epi)
cv2.imwrite("results/epi/vertical_epi.png", vertical_epi)

print("🎉 EPIs generated from individual frames → saved in results/epi/")
