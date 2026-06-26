import os, glob, shutil

# Base directories
SRC_DIR = "/mnt/d/thesisProject/LightFieldGrid/levelone/results/volumes"
AUG_DIR = os.path.join(SRC_DIR, "augmented")   # ✅ fixed line
OUT_DIR = os.path.join(SRC_DIR, "train_ready")

# Create output folders
os.makedirs(OUT_DIR, exist_ok=True)

for cls in ["orange", "cup"]:
    cls_out = os.path.join(OUT_DIR, cls)
    os.makedirs(cls_out, exist_ok=True)

    # 🔹 Copy original volume (the main .npy)
    main_path = os.path.join(SRC_DIR, f"{cls}_epi.npy")
    if os.path.exists(main_path):
        shutil.copy(main_path, os.path.join(cls_out, f"{cls}_orig.npy"))
        print(f"✅ Added original: {cls}_orig.npy")
    else:
        print(f"⚠️ Original {cls}_epi.npy not found!")

    # 🔹 Copy all augmented volumes
    aug_files = sorted(glob.glob(os.path.join(AUG_DIR, cls, "*.npy")))
    for a in aug_files:
        shutil.copy(a, cls_out)

    print(f"✅ Copied {len(aug_files)} augmented files for {cls}")

print("\n🎯 Training dataset ready at:", OUT_DIR)
