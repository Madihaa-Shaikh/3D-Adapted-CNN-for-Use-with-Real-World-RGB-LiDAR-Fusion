# StepSevenOneLightFieldVolumeDataset.py
# -------------------------------------
# Stage 8.1 — Light-Field Volume Dataset (for 3D reconstruction)
# Loads class-wise volumetric .npy files and prepares tensors for 3D CNN/LSTM.

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LightFieldVolumeDataset(Dataset):
    """
    Loads volumetric light-field data (.npy) for 3D reconstruction / autoencoding.
    Each sample is a 3D tensor (D, H, W) created from stacked EPIs, then:
      - normalized (zero-mean, unit-std)
      - downsampled (stride sampling) for memory efficiency
      - channel dimension added -> (C, D, H, W) with C=1
    For reconstruction tasks, input == target.
    """
    def __init__(self, data_root: str):
        super().__init__()
        self.data_root = data_root
        self.samples = []  # list of absolute paths to .../lightfield_volume.npy

        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        # Collect all volumes from each class directory
        for cls in sorted(os.listdir(self.data_root)):
            cls_path = os.path.join(self.data_root, cls)
            if not os.path.isdir(cls_path):
                continue

            # Each sample is a subfolder (e.g., sample_01, sample_02, ...)
            for sample_dir in sorted(os.listdir(cls_path)):
                sample_path = os.path.join(cls_path, sample_dir)
                vol_path = os.path.join(sample_path, "lightfield_volume.npy")
                if os.path.exists(vol_path):
                    self.samples.append(vol_path)

        print(f"✅ Found {len(self.samples)} volumetric samples in {self.data_root}")

    def __len__(self):
        return len(self.samples)

    def _preprocess(self, volume: np.ndarray) -> np.ndarray:
        """Normalize, downsample, and add channel dimension."""
        # 1) normalize per-sample
        volume = volume.astype(np.float32)
        volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-5)

        # 2) downsample (tuned to your earlier training setup)
        #    original was (D, H, W) ~ (20, 5, 2464)
        #    keep all D; sample H by 2, W by 4 -> (20, ~3, ~616)
        volume = volume[:, ::2, ::4]

        # 3) add channel -> (C, D, H, W) with C=1
        volume = np.expand_dims(volume, axis=0)
        return volume

    def __getitem__(self, idx: int):
        vol_path = self.samples[idx]
        volume = np.load(vol_path)  # shape: (D, H, W)
        volume = self._preprocess(volume)  # shape: (1, D, H, W)

        # For reconstruction/autoencoder: input == target
        x = torch.from_numpy(volume)          # float32 tensor
        y = torch.from_numpy(volume.copy())   # target same as input
        return x, y


# ----------------------------------------------------------------------
# Sanity run (only executes when you run this file directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    DATA_ROOT = "./results/data_lightfield"

    dataset = LightFieldVolumeDataset(DATA_ROOT)
    print(f"Total volumetric samples loaded: {len(dataset)}")

    if len(dataset) == 0:
        print("⚠️ No samples found. Check folder structure:")
        print("  ./results/data_lightfield/<class_name>/<sample_x>/lightfield_volume.npy")
        raise SystemExit(0)

    # Peek 1 item
    x, y = dataset[0]
    print("Single sample tensor shapes:")
    print("  input (x): ", tuple(x.shape))  # (1, D, H, W)
    print("  target(y): ", tuple(y.shape))

    # Basic stats
    print("min/max/mean (x): ",
          float(x.min().item()),
          float(x.max().item()),
          float(x.mean().item()))

    # Try a small DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    xb, yb = next(iter(loader))
    print("Batch shapes from DataLoader:")
    print("  xb: ", tuple(xb.shape))  # (B, 1, D, H, W)
    print("  yb: ", tuple(yb.shape))

    print("✅ Dataset loader looks good. Ready for Stage 8.2 (3D CNN Encoder).")
