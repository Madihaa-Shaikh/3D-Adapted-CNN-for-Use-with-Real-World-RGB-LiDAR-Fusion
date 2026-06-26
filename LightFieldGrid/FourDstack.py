import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

epi_folder = "./results/epi"

# 🔹 Load all EPI images
epi_files = sorted([
    f for f in os.listdir(epi_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

if not epi_files:
    raise FileNotFoundError(f"❌ No EPI slices found in {epi_folder}")

print(f"✅ Found {len(epi_files)} EPI slices")

epi_slices = []
target_size = None

for f in epi_files:
    path = os.path.join(epi_folder, f)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Skipping unreadable file: {f}")
        continue

    # Set reference size based on first image
    if target_size is None:
        target_size = (img.shape[1], img.shape[0])  # (width, height)
        print(f"📏 Reference size: {target_size}")

    # Resize all to the same shape
    img_resized = cv2.resize(img, target_size)
    epi_slices.append(img_resized)

# 🔹 Stack into 3D volume
volume = np.stack(epi_slices, axis=0)
print(f"📦 3D volume shape: {volume.shape}  → (Depth, Height, Width)")

# 🔹 Normalize for CNN
volume = volume.astype(np.float32) / 255.0

# 🔹 Save as NumPy file
output_path = "./results/epi/lightfield_volume.npy"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, volume)
print(f"💾 Saved 3D EPI volume → {output_path}")

# 🔹 Quick visualization
plt.figure(figsize=(10, 4))
for i in range(min(3, len(volume))):
    plt.subplot(1, 3, i + 1)
    plt.imshow(volume[i], cmap='jet')
    plt.title(f"Slice {i}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("results/epi/volume_preview.png")
print("📸 Saved EPI preview → results/epi/volume_preview.png")
