# StepSevenTwo_Encoder3D.py
# -------------------------
# Stage 8.2 — 3D CNN Encoder
# Input : (B, 1, D, H, W)  e.g., (B, 1, 20, 3, 616)
# Output: (B, T, F) sequence for LSTM, e.g., (B, 128, 128)

import torch
import torch.nn as nn
import torch.nn.functional as F

# (Optional) import your dataset to smoke-test the encoder
from StepSevenOneLightFieldVolumeDataset import LightFieldVolumeDataset
from torch.utils.data import DataLoader

class CNN3D_Encoder(nn.Module):
    """
    3D CNN encoder that extracts spatial–depth features from light-field volumes.
    We aggressively downsample (by design) along W and D, but keep H stable.
    AdaptiveAvgPool3d → fixed-size latent grid, then reshape to (B, T, F).
    """
    def __init__(self, out_channels=128, seq_D=4, seq_H=1, seq_W=32):
        super().__init__()
        self.features = nn.Sequential(
            # (B, 1, D, H, W)
            nn.Conv3d(1, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),

            # pool along D and W, keep H (height) intact
            nn.MaxPool3d(kernel_size=(2, 1, 2)),  # (D/2, H, W/2)

            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),

            # Fix to a compact latent grid regardless of input size
            nn.AdaptiveAvgPool3d((seq_D, seq_H, seq_W))  # -> (B, C=out_channels, seq_D, seq_H, seq_W)
        )
        self.out_channels = out_channels
        self.seq_D = seq_D
        self.seq_H = seq_H
        self.seq_W = seq_W

    def forward(self, x):
        """
        x: (B, 1, D, H, W)
        returns seq: (B, T, F)  where T = seq_D * seq_H * seq_W, F = out_channels
        """
        z = self.features(x)  # (B, C, seq_D, seq_H, seq_W)
        B, C, d, h, w = z.shape
        # rearrange to sequence for LSTM: (B, T, F)
        z = z.permute(0, 2, 3, 4, 1).contiguous()     # (B, d, h, w, C)
        z = z.view(B, d * h * w, C)                   # (B, T, F)
        return z


# ----------------------------------------------------------------------
# Smoke test when running this file directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    DATA_ROOT = "./results/data_lightfield"

    # 1) load a small batch from dataset (uses your Step 8.1 loader)
    ds = LightFieldVolumeDataset(DATA_ROOT)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    x, _ = next(iter(dl))  # x: (B, 1, D, H, W)
    print("Input batch shape:", tuple(x.shape))

    # 2) build encoder and run forward
    encoder = CNN3D_Encoder(out_channels=128, seq_D=4, seq_H=1, seq_W=32)
    with torch.no_grad():
        z = encoder(x)  # (B, T, F)

    print("Encoded sequence shape:", tuple(z.shape))
    B, T, F = z.shape
    print(f"B={B}, T={T} (={encoder.seq_D}*{encoder.seq_H}*{encoder.seq_W}), F={F} (=out_channels)")

    # Check a single timestep embedding stats
    print("z min/max/mean:", float(z.min()), float(z.max()), float(z.mean()))
    print("✅ Encoder smoke-test OK — ready for Stage 8.3 (LSTM Decoder).")
