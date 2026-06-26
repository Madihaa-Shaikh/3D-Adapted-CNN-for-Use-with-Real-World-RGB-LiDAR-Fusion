import numpy as np
import os
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

# -----------------------
# Paths
# -----------------------
IN_PATH = "./results/output_StepNineVisualization/output_volume.npy"
OUT_DIR = "./results/output_StepElevenPLY"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Load & Normalize Volume
# -----------------------
v = np.load(IN_PATH).astype(np.float32)
print(f"✅ Loaded reconstructed volume: {v.shape} min: {v.min()} max: {v.max()}")

# Normalize [0,1]
v = (v - v.min()) / (v.max() - v.min() + 1e-8)

# Boost contrast
v = v ** 0.5
v[v < 0.4] = 0.0  # remove faint noise

# -----------------------
# Coordinate Grid (MATCHING AXIS ORDER)
# -----------------------
# -----------------------
# Coordinate Grid with scaling correction
# -----------------------
D, H, W = v.shape  # (20, 3, 616)
scale_z, scale_y, scale_x = 5.0, 40.0, 1.0  # exaggerate depth and height

z, y, x = np.meshgrid(
    np.arange(D) * scale_z,
    np.arange(H) * scale_y,
    np.arange(W) * scale_x,
    indexing='ij'
)


# Verify shapes
print("Coordinate grids:", x.shape, y.shape, z.shape, "| Volume:", v.shape)

# -----------------------
# Apply mask and stack points
# -----------------------
mask = v > 0.4
points = np.column_stack((x[mask], y[mask], z[mask]))
intensity = v[mask]

print(f"Extracted {len(points)} visible voxels")

# -----------------------
# Color Mapping
# -----------------------
colors = (plt.cm.plasma(intensity)[:, :3] * 255).astype(np.uint8)

vertex = np.array(
    [(p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(points, colors)],
    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
           ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
)

ply = PlyData([PlyElement.describe(vertex, 'vertex')])
ply_path = os.path.join(OUT_DIR, "reconstruction_voxel_scaled.ply")
ply.write(ply_path)

print(f"💾 Saved voxel point-cloud → {ply_path}")
print(f"🧩 Visible voxels: {len(points)}")
print("✅ Done — open the PLY in MeshLab, Blender, or Meshy.ai")
