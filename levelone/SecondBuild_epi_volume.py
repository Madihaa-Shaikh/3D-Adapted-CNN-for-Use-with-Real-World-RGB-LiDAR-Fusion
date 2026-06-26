# Each folder (orange / cup) has ~29 frames
# Create Light-Field EPI Volume (D,H,W)
import numpy as np, cv2, glob, os

def build_epi_volume(input_dir, out_path):
    imgs = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    views = []
    for f in imgs:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128,128))
        views.append(img)
    V = np.stack(views, axis=0).astype(np.float32)
    V = (V - V.min()) / (V.max() - V.min())  # normalize 0-1
    np.save(out_path, V)
    print(f" Saved volume: {V.shape} → {out_path}")

build_epi_volume(
      # "/mnt/d/thesisProject/LightFieldGrid/levelone/frames/orange",
      # "/mnt/d/thesisProject/LightFieldGrid/levelone/results/volumes/orange_epi.npy"
       "/mnt/d/thesisProject/LightFieldGrid/levelone/frames/cup",
      "/mnt/d/thesisProject/LightFieldGrid/levelone/results/volumes/cup_epi.npy"
)
