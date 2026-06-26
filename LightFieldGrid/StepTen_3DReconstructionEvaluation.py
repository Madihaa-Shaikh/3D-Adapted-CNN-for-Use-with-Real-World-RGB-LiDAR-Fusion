# Stage 10 — Evaluation & Visualization Report
# - Compares Input vs Reconstructed volumes
# - Computes MSE + SSIM
# - Generates side-by-side figure (Input vs Output vs Difference)
# - Optionally interactive 3D visualization with Plotly

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

# -----------------------
# Config
# -----------------------
RESULTS_DIR = "./results/output_StepNineVisualization"
OUT_DIR = "./results/output_StepTenEvaluation"
os.makedirs(OUT_DIR, exist_ok=True)

input_path  = os.path.join(RESULTS_DIR, "input_volume.npy")
output_path = os.path.join(RESULTS_DIR, "output_volume.npy")

# -----------------------
# Load Volumes
# -----------------------
x = np.load(input_path)
y = np.load(output_path)
print(f"✅ Loaded volumes → input {x.shape}, reconstructed {y.shape}")

# Normalize [0,1]
x = (x - x.min()) / (x.max() - x.min() + 1e-8)
y = (y - y.min()) / (y.max() - y.min() + 1e-8)

# -----------------------
# Metrics
# -----------------------
mse = mean_squared_error(x.flatten(), y.flatten())
ssim_val = ssim(
    x.mean(axis=0),
    y.mean(axis=0),
    data_range=1.0,
    win_size=3,         # smaller window for thin slices
    channel_axis=None   # grayscale input
)

print(f"📊 MSE: {mse:.6f}   SSIM: {ssim_val:.4f}")

# -----------------------
# Visualization (slices) 
# -----------------------
D, H, W = x.shape
mid_d, mid_w = D//2, W//2

diff = np.abs(x - y)
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(x[mid_d], cmap='inferno')
plt.title("Input slice (D mid)")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(y[mid_d], cmap='inferno')
plt.title("Reconstructed slice (D mid)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(diff[mid_d], cmap='magma')
plt.title("|Difference| (Error Map)")
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "input_output_difference.png"), dpi=200)
plt.close()
print("🖼 Saved → input_output_difference.png")

# -----------------------
# Optional 3D Visualization with Plotly
# -----------------------
import scipy.ndimage as nd

# Upscale the very thin dimension (H=3 → ~60)
print("🔧 Upscaling reconstructed volume for visualization...")
y_scaled = nd.zoom(y, (1, 20, 1))   # depth×height×width = (20, 60, 616)
y_scaled = (y_scaled - y_scaled.min()) / (y_scaled.max() - y_scaled.min() + 1e-5)
y_scaled[y_scaled < 0.3] = 0
y_scaled[y_scaled > 1.0] = 1.0
print("✅ Scaled shape for display:", y_scaled.shape)


def plot_3d_voxel(volume, title, outfile):
    """
    Render 3D structure using Plotly
    """
    vol = (volume > 0.1).astype(np.float32)   # lower threshold
    D, H, W = vol.shape
    fig = go.Figure(data=go.Volume(
    x=np.tile(np.arange(W), H*D),
    y=np.repeat(np.arange(H), W*D),
    z=np.repeat(np.arange(D), H*W),
    value=vol.flatten(),
    opacity=0.35,           # more visible
    surface_count=40,       # smoother contours
    isomin=0.05,            # include faint voxels
    isomax=1.0,
    colorscale='Inferno'
))
    fig.update_layout(
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='manual',
        aspectratio=dict(x=2, y=1, z=1)
    ),
    title=title
)

    fig.update_layout(scene=dict(aspectmode='cube'),
                      title=title)
    fig.write_html(outfile)
    print(f"💾 Saved interactive 3D visualization → {outfile}")
    # -----------------------
# Optional Upscaling + Intensity Amplification for Visualization Only
# -----------------------
import scipy.ndimage as nd

print("🔧 Upscaling + amplifying reconstructed volume for visualization...")

# Upscale thin dimension (3 → 60)
y_scaled = nd.zoom(y, (1, 20, 1))   # (20, 60, 616)

# Normalize to [0,1]
y_scaled = (y_scaled - y_scaled.min()) / (y_scaled.max() - y_scaled.min() + 1e-5)

# 🔹 Boost weak intensities so faint structures become visible
y_scaled = np.power(y_scaled, 0.6)     # gamma correction (brightens mid-tones)
y_scaled *= 2.5                        # intensity boost
y_scaled[y_scaled > 1.0] = 1.0

# 🔹 Apply soft threshold
y_scaled[y_scaled < 0.15] = 0

print("✅ Scaled & amplified shape for display:", y_scaled.shape,
      "| min:", y_scaled.min(), "max:", y_scaled.max(), "mean:", y_scaled.mean())

plot_3d_voxel(y_scaled, "Reconstructed 3D Volume (Upscaled)", os.path.join(OUT_DIR, "reconstructed_3d.html"))

print("✅ Stage 10 complete! Quantitative + qualitative report ready.")
