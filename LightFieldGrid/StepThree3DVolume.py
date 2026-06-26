# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt

# # ===========================
# # Step 1: Define folders
# # ===========================
# input_folder = "./results/epi/output_StepTwoGenerateEPISlicesFromL"
# output_folder = "./results/epi/output_StepThree3DVolume"
# os.makedirs(output_folder, exist_ok=True)

# # ===========================
# # Step 2: Load all EPIs
# # ===========================
# epi_files = sorted([
#     f for f in os.listdir(input_folder)
#     if f.lower().endswith((".png", ".jpg", ".jpeg"))
# ])

# if not epi_files:
#     raise FileNotFoundError(f"❌ No EPI images found in {input_folder}")

# print(f"✅ Found {len(epi_files)} EPI slices to combine → {input_folder}")

# # ===========================
# # Step 3: Read & resize EPIs
# # ===========================
# epi_slices = []
# target_size = None

# for f in epi_files:
#     path = os.path.join(input_folder, f)
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print(f"⚠️ Skipping unreadable file: {f}")
#         continue

#     if target_size is None:
#         target_size = (img.shape[1], img.shape[0])
#         print(f"📏 Reference EPI size: {target_size}")

#     img_resized = cv2.resize(img, target_size)
#     epi_slices.append(img_resized)

# # ===========================
# # Step 4: Stack into 3D Volume
# # ===========================
# volume = np.stack(epi_slices, axis=0)
# volume = volume.astype(np.float32) / 255.0

# print(f"📦 Created Light Field 3D Volume → shape: {volume.shape} (Depth, Height, Width)")

# # Save NumPy volume
# np.save(os.path.join(output_folder, "lightfield_volume.npy"), volume)
# print(f"💾 Saved 3D volume → {output_folder}/lightfield_volume.npy")

# # ===========================
# # Step 5: Quick visualization
# # ===========================
# plt.figure(figsize=(10, 4))
# for i in range(min(3, len(volume))):
#     plt.subplot(1, 3, i + 1)
#     plt.imshow(volume[i], cmap='jet')
#     plt.title(f"Slice {i}")
#     plt.axis('off')

# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, "lightfield_volume_preview.png"))
# print(f"📸 Saved Light Field Volume Preview → {output_folder}/lightfield_volume_preview.png")
import numpy as np

volume = np.load("./results/epi/output_StepThree3DVolume/lightfield_volume.npy")
print("✅ Volume loaded:", volume.shape)
print("🔹 Min:", volume.min(), " Max:", volume.max(), " Mean:", volume.mean())
