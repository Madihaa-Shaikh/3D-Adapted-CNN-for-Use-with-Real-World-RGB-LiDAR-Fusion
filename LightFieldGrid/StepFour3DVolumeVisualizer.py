import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# Load your 3D Light Field volume
volume = np.load("./results/epi/output_StepThree3DVolume/lightfield_volume.npy")
print("✅ Volume loaded:", volume.shape)

# Normalize
vmin, vmax = volume.min(), volume.max()
volume_norm = (volume - vmin) / (vmax - vmin + 1e-8)

# Downsample a bit for visualization (to speed up)
volume_small = volume_norm[::2, :, ::20]  # reduce resolution
D, H, W = volume_small.shape

# Prepare coordinates
z, y, x = np.indices((D, H, W))
values = volume_small

# Threshold for visible voxels
threshold = 0.55
mask = values > threshold

# 3D visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.voxels(mask, facecolors=plt.cm.jet(values), edgecolor="k", linewidth=0.2)

ax.set_xlabel("Width (x)")
ax.set_ylabel("Height (y)")
ax.set_zlabel("Depth (EPI index)")
ax.set_title("3D Light Field Volume Visualization")
plt.tight_layout()
plt.savefig("./results/epi/output_StepThree3DVolume/lightfield_volume_voxels.png", dpi=300)
plt.show()

print("📸 Saved 3D voxel visualization → ./results/output_StepThree3DVolume/lightfield_volume_voxels.png")
