import os
import numpy as np
from scipy.ndimage import rotate
import random

# ==========================
# Configuration
# ==========================
DATA_ROOT = "./results/data_lightfield"
AUG_PER_SAMPLE = 10  # how many new versions to make for each volume
SAVE_SUFFIX = "_aug"  # name pattern for augmented files

os.makedirs(DATA_ROOT, exist_ok=True)

# ==========================
# Helper functions
# ==========================
def augment_volume(volume):
    """Apply random augmentations"""
    v = volume.copy()
    if random.random() > 0.5:
        v = np.flip(v, axis=-1).copy()  # horizontal flip
    if random.random() > 0.5:
        v = np.flip(v, axis=-2).copy()  # vertical flip
    if random.random() > 0.5:
        angle = random.choice([-10, -5, 5, 10])
        v = rotate(v, angle, axes=(-2, -1), reshape=False)
    if random.random() > 0.5:
        factor = 1.0 + (0.1 * np.random.randn())  # ±10% brightness
        v *= factor
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.01, size=v.shape)
        v += noise
    return v.astype(np.float32)


# ==========================
# Generate Augmentations
# ==========================
for cls in os.listdir(DATA_ROOT):
    cls_path = os.path.join(DATA_ROOT, cls)
    if not os.path.isdir(cls_path):
        continue

    for sample_dir in os.listdir(cls_path):
        sample_path = os.path.join(cls_path, sample_dir)
        vol_path = os.path.join(sample_path, "lightfield_volume.npy")
        if not os.path.exists(vol_path):
            continue

        volume = np.load(vol_path)
        print(f"🔹 Augmenting {vol_path}")

        for i in range(1, AUG_PER_SAMPLE + 1):
            aug_volume = augment_volume(volume)
            save_path = os.path.join(sample_path, f"lightfield_volume{SAVE_SUFFIX}{i:02d}.npy")
            np.save(save_path, aug_volume)

print("✅ Dataset augmentation complete! Check new *_augXX.npy files inside each sample folder.")
