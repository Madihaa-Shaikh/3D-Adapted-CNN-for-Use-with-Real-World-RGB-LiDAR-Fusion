import cv2
import numpy as np
import os

# 🔹 Path to your grid image
grid_path = "./results/lightfield_grid.png"

# 🔹 Angular grid size (rows × cols)
U, V = 5, 6   # <-- adjust if your grid layout differs

# 🔹 Load grid image
grid_img = cv2.imread(grid_path)
if grid_img is None:
    raise FileNotFoundError(f"❌ Could not read {grid_path}")

H_total, W_total, C = grid_img.shape
print(f"Grid image size: {W_total}x{H_total}")

# 🔹 Calculate size of each sub-aperture image
H_each = H_total // U
W_each = W_total // V
print(f"Each view size: {W_each}x{H_each}")

# 🔹 Create 4D light field array (U, V, H_each, W_each, 3)
L = np.zeros((U, V, H_each, W_each, C), dtype=np.uint8)

# 🔹 Extract sub-images
for u in range(U):
    for v in range(V):
        y0 = u * H_each
        y1 = (u + 1) * H_each
        x0 = v * W_each
        x1 = (v + 1) * W_each
        L[u, v, :, :, :] = grid_img[y0:y1, x0:x1, :]

print(f"✅ Light field array created with shape: {L.shape}")
np.save("./results/light_field_array.npy", L)
print("💾 Saved 4D Light Field as ./results/light_field_array.npy")
