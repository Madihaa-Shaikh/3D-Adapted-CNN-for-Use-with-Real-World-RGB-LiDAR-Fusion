import numpy as np
from pathlib import Path

# ============================
# CONFIG
# ============================

ROOT = Path("/mnt/d/thesisProject/LightFieldGrid/levelone/KittiDataset/Kitti/processed")

CLASSES = ["car", "pedestrian", "cyclist"]
VOXEL_SIZE = 32  # 32x32x32 voxels

OUTPUT_DIR = Path("/mnt/d/thesisProject/LightFieldGrid/levelone/KittiDataset/Kitti/voxels")
OUTPUT_DIR.mkdir(exist_ok=True)



# ============================
# VOXELIZATION FUNCTION
# ============================

def pointcloud_to_voxels(pc, grid=32):
    if len(pc) == 0:
        return np.zeros((grid, grid, grid), dtype=np.uint8)

    # Normalize to [0,1]
    mins = pc.min(axis=0)
    maxs = pc.max(axis=0)
    ranges = maxs - mins + 1e-6
    pc_norm = (pc - mins) / ranges  

    # Convert to voxel indices
    idx = (pc_norm * (grid - 1)).astype(int)

    # Create voxel grid
    voxels = np.zeros((grid, grid, grid), dtype=np.uint8)
    voxels[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    return voxels



# ============================
# MAIN LOOP
# ============================

for cls in CLASSES:

    input_dir = ROOT / cls
    output_class_dir = OUTPUT_DIR / cls
    output_class_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npy"))
    print(f"\nProcessing class: {cls}  ({len(files)} samples)")

    for f in files:

        pc = np.load(f)

        # Convert to voxel grid
        vox = pointcloud_to_voxels(pc, VOXEL_SIZE)

        # Save as .npz (compressed)
        out_path = output_class_dir / (f.stem + ".npz")
        np.savez_compressed(out_path, voxels=vox)

        print("Saved voxel:", out_path)

print("\n✅ DONE — All point clouds converted to voxel grids.")
