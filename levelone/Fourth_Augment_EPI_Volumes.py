import os, glob
import numpy as np
import cv2

# -----------------------------
# PATH CONFIGURATION
# -----------------------------
SRC_DIR = "/mnt/d/thesisProject/LightFieldGrid/levelone/results/volumes"
OUT_DIR = os.path.join(SRC_DIR, "augmented")
os.makedirs(OUT_DIR, exist_ok=True)

AUG_MULTIPLIER = 15  # number of augmented samples per original
np.random.seed(42)


# -----------------------------
# AUGMENTATION FUNCTION
# -----------------------------
def augment_volume(V):
    """Safe augmentations that preserve EPI structure"""
    out = V.copy().astype(np.float32)

    # ensure volume is normalized 0–255 before processing
    if out.max() <= 1.0:
        out = out * 255.0

    # random horizontal flip
    if np.random.rand() > 0.5:
        out = np.flip(out, axis=2)

    # random small rotation
    angle = np.random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((V.shape[2] // 2, V.shape[1] // 2), angle, 1.0)
    rotated = [cv2.warpAffine(out[i], M, (V.shape[2], V.shape[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
               for i in range(V.shape[0])]
    out = np.stack(rotated, axis=0)

    # light brightness / contrast jitter
    alpha = np.random.uniform(0.9, 1.1)
    beta  = np.random.uniform(-10, 10)
    out = out * alpha + beta

    # tiny Gaussian noise
    noise = np.random.normal(0, 1.5, out.shape)
    out = out + noise

    # clip safely and normalize back to 0–1
    out = np.clip(out, 0, 255)
    out = out / 255.0

    return out.astype(np.float32)

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    for cls in ["orange", "cup"]:
        in_path = os.path.join(SRC_DIR, cls, f"{cls}_epi.npy")  # ✅ FIXED PATH
        if not os.path.exists(in_path):
            print(f"⚠️ {cls}_epi.npy not found at {in_path}")
            continue

        V = np.load(in_path)
        print(f"Loaded {cls}: {V.shape}")

        out_cls_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(out_cls_dir, exist_ok=True)

        for i in range(AUG_MULTIPLIER):
            aug_V = augment_volume(V)
            out_path = os.path.join(out_cls_dir, f"{cls}_aug{i}.npy")
            np.save(out_path, aug_V)
            print(f"✅ Saved: {out_path}")

    print("\n🎯 Augmentation completed successfully!")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
