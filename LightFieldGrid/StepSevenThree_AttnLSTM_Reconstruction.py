# Stage 8.3 + 8.4 + 8.5
# BiLSTM + Multi-Head Attention + 3D Reconstruction Head (with full training loop)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 🔹 import your Stage 8.1 and 8.2 modules
from StepSevenOneLightFieldVolumeDataset import LightFieldVolumeDataset
from StepSevenTwo_Encoder3D import CNN3D_Encoder

# -----------------------
# Config
# -----------------------
DATA_ROOT = "./results/data_lightfield"
SAVE_DIR  = "./results/output_StepEight3DReconstruction"
BATCH_SIZE = 2
EPOCHS = 40
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)
print("💻 Using device:", DEVICE)

# -----------------------
# Attention + BiLSTM block
# -----------------------
class AttnBiLSTM(nn.Module):
    """
    Input : seq (B, T, F)
    Output: seq' (B, T, H) with contextualized embeddings
    """
    def __init__(self, in_dim=128, hid=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.bilstm = nn.LSTM(in_dim, hid // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.norm1  = nn.LayerNorm(hid)

        self.attn   = nn.MultiheadAttention(embed_dim=hid, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2  = nn.LayerNorm(hid)

        self.ff     = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hid, hid)
        )
        self.norm3  = nn.LayerNorm(hid)

    def forward(self, seq):
        # BiLSTM
        z, _ = self.bilstm(seq)          # (B, T, H)
        z = self.norm1(z)

        # Self-attention (residual)
        z2, _ = self.attn(z, z, z)       # (B, T, H)
        z = self.norm2(z + z2)

        # Feed-forward (residual)
        z3 = self.ff(z)
        z = self.norm3(z + z3)
        return z                          # (B, T, H)

# -----------------------
# Reconstruction Head
# -----------------------
class ReconHead3D(nn.Module):
    """
    Takes sequence features (B, T, H) → global latent → 3D small grid → conv refine → upsample to target (D,H,W).
    """
    def __init__(self, seq_hid=256, base_ch=64, latent_d=4, latent_h=1, latent_w=32):
        super().__init__()
        self.latent_shape = (base_ch, latent_d, latent_h, latent_w)
        flat_latent = base_ch * latent_d * latent_h * latent_w

        self.to_latent = nn.Sequential(
            nn.Linear(seq_hid, 512), nn.ReLU(inplace=True),
            nn.Linear(512, flat_latent), nn.ReLU(inplace=True),
        )

        # light 3D refinement stack (doesn't upsample much; upsampling done by interpolate)
        self.refine = nn.Sequential(
            nn.Conv3d(base_ch, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, 1),              # → (B,1,d,h,w)
            nn.Sigmoid()
        )

    def forward(self, seq_ctx, target_shape):
        """
        seq_ctx: (B, T, H) context features
        target_shape: tuple (D, H, W) of desired output
        """
        B = seq_ctx.size(0)

        # Global pooling over time (mean). Optionally try max/attention pooling variants.
        global_code = seq_ctx.mean(dim=1)          # (B, H)

        # map to latent 3D grid
        latent = self.to_latent(global_code)       # (B, flat_latent)
        C, d, h, w = self.latent_shape
        latent = latent.view(B, C, d, h, w)        # (B, C, d, h, w)

        # refine by small conv stack
        small_vol = self.refine(latent)            # (B, 1, d, h, w)

        # upsample to target
        D_out, H_out, W_out = target_shape
        out = F.interpolate(small_vol, size=(D_out, H_out, W_out), mode="trilinear", align_corners=False)
        return out                                  # (B,1,D_out,H_out,W_out)

# -----------------------
# Full Model
# -----------------------
class LightFieldReconstructor(nn.Module):
    """
    Encoder (3D CNN) → AttnBiLSTM → 3D Reconstruction Head
    """
    def __init__(self):
        super().__init__()
        self.encoder = CNN3D_Encoder(out_channels=128, seq_D=4, seq_H=1, seq_W=32)
        self.temporal = AttnBiLSTM(in_dim=128, hid=256, num_heads=4, dropout=0.1)
        self.recon =  ReconHead3D(seq_hid=256, base_ch=64, latent_d=4, latent_h=1, latent_w=32)

    def forward(self, x, target_shape):
        """
        x: (B,1,D,H,W)
        target_shape: (D,H,W) from the batch target
        """
        seq = self.encoder(x)                  # (B, T=128, F=128)
        seq_ctx = self.temporal(seq)           # (B, T=128, H=256)
        yhat = self.recon(seq_ctx, target_shape)  # (B,1,D,H,W)
        return yhat

# -----------------------
# Training
# -----------------------
def train():
    # data
    ds = LightFieldVolumeDataset(DATA_ROOT)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = LightFieldReconstructor().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    # MSE for voxel similarity (values are in [0,1] after Sigmoid)
    criterion = nn.MSELoss()

    print("✅ Model ready: Encoder + BiLSTM+Attention + 3D Decoder")

    best = 1e9
    for epoch in range(1, EPOCHS+1):
        model.train()
        run_loss = 0.0

        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)        # xb/yb: (B,1,D,H,W)
            target_shape = yb.shape[-3:]                 # (D,H,W)

            opt.zero_grad()
            pred = model(xb, target_shape)               # (B,1,D,H,W)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()

            run_loss += loss.item()

        epoch_loss = run_loss / max(1, len(dl))
        if epoch_loss < best:
            best = epoch_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "3DRecon_AttnBiLSTM_best.pt"))

        print(f"[Epoch {epoch:02d}/{EPOCHS}] Loss: {epoch_loss:.6f}  (best {best:.6f})")

    print(f"💾 Saved best model → {os.path.join(SAVE_DIR, '3DRecon_AttnBiLSTM_best.pt')}")

if __name__ == "__main__":
    train()
