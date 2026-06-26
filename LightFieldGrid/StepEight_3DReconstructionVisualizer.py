# Stage 9 — 3D Reconstruction Visualization
# - Loads best recon model
# - Runs prediction on one sample from your dataset
# - Saves slice visualizations and optional point cloud (.ply)

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# ✅ Use your Stage-8 modules
from StepSevenOneLightFieldVolumeDataset import LightFieldVolumeDataset
from StepSevenThree_AttnLSTM_Reconstruction import LightFieldReconstructor  # model class

# -----------------------
# Config
# -----------------------
DATA_ROOT  = "./results/data_lightfield"
MODEL_PATH = "./results/output_StepEight3DReconstruction/3DRecon_AttnBiLSTM_best.pt"
OUT_DIR    = "./results/output_StepNineVisualization"
BATCH_SIZE = 1
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)
print("💻 Using device:", DEVICE)

# -----------------------
# Utility: save point cloud as .ply (view in MeshLab/CloudCompare)
# -----------------------
def save_point_cloud_ply(volume_3d, ply_path, threshold=0.4, voxel_scale=(1.0,1.0,1.0)):
    """
    volume_3d: numpy (D, H, W) in [0,1]
    threshold: prob > threshold -> keep as point
    voxel_scale: spacing for (z,y,x)
    """
    D, H, W = volume_3d.shape
    zz, yy, xx = np.nonzero(volume_3d > threshold)
    if len(zz) == 0:
        print("⚠️ No points above threshold — try lowering it.")
        return

    # Scale to real units if needed
    sx, sy, sz = voxel_scale[2], voxel_scale[1], voxel_scale[0]
    pts = np.stack([xx*sx, yy*sy, zz*sz], axis=1)

    # (Optional) simple grey color from volume value
    colors = (volume_3d[zz, yy, xx] * 255).clip(0,255).astype(np.uint8)
    cols = np.stack([colors, colors, colors], axis=1)  # (N,3)

    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x,y,z),(r,g,b) in zip(pts, cols):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
    print(f"💾 Saved point cloud → {ply_path}  (points: {pts.shape[0]})")

# -----------------------
# Visualization helpers
# -----------------------
def save_volume_slices(vol, tag, out_dir):
    """
    vol: numpy (D,H,W) in [0,1]
    Saves middle slices along D and W for quick inspection.
    """
    D, H, W = vol.shape
    mid_d = D // 2
    mid_w = W // 2

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(vol[mid_d], cmap="inferno", aspect="auto")
    plt.title(f"{tag} • slice @ D={mid_d}")
    plt.axis("off")

    plt.subplot(1,2,2)
    # slice along W → take all D x H at W=mid_w (transpose for nicer view)
    plt.imshow(vol[:,:,mid_w].T, cmap="inferno", aspect="auto")
    plt.title(f"{tag} • slice @ W={mid_w}")
    plt.axis("off")

    fname = os.path.join(out_dir, f"{tag.replace(' ','_').lower()}_slices.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"📸 Saved slices → {fname}")

# -----------------------
# Main
# -----------------------
def main():
    # 1) Load dataset (same preprocessing as Stage-8.1)
    ds = LightFieldVolumeDataset(DATA_ROOT)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2) Load trained recon model
    model = LightFieldReconstructor().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded model weights from {MODEL_PATH}")

    # 3) Take one sample
    xb, yb = next(iter(dl))       # xb/yb: (B,1,D,H,W)
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    target_shape = yb.shape[-3:]  # (D,H,W)

    # 4) Predict
    with torch.no_grad():
        pred = model(xb, target_shape)          # (B,1,D,H,W)
        pred = torch.clamp(pred, 0, 1)

    inp = xb[0,0].detach().cpu().numpy()        # (D,H,W)
    gt  = yb[0,0].detach().cpu().numpy()        # (D,H,W) same as inp
    out = pred[0,0].detach().cpu().numpy()      # (D,H,W)

    # 5) Save numpy volumes (optional)
    np.save(os.path.join(OUT_DIR, "input_volume.npy"),  inp)
    np.save(os.path.join(OUT_DIR, "output_volume.npy"), out)
    print("💾 Saved input/output volumes (.npy)")

    # 6) Slice visualizations (input vs output)
    save_volume_slices(inp, "Input Volume",  OUT_DIR)
    save_volume_slices(out, "Output Reconstructed", OUT_DIR)

    # 7) Point cloud export (.ply)
    #   - Use slightly higher threshold to get cleaner shape; tune as needed
    save_point_cloud_ply(out, os.path.join(OUT_DIR, "reconstruction.ply"),
                         threshold=0.5, voxel_scale=(1.0, 1.0, 1.0))

    print("✅ Stage 9 visualization complete.")

if __name__ == "__main__":
    main()
