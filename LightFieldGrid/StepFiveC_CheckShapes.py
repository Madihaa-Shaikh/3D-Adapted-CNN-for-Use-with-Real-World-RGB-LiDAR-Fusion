import os, numpy as np, cv2, torch

# -----------------------------
# 1️⃣ Check light-field data shapes
# -----------------------------
paths = {
    "4D light field": "./results/light_field_array.npy",  # (U,V,H,W,C)
    "3D volume": "./results/epi/output_StepThree3DVolume/lightfield_volume.npy",  # (D,H,W)
}

for name, p in paths.items():
    if os.path.exists(p):
        arr = np.load(p, allow_pickle=False)
        print(f"\n✅ {name} found at: {p}")
        print(f"📏 Shape: {arr.shape}")
        print("-" * 50)
    else:
        print(f"⚠️ Missing: {name} not found at {p}")

# -----------------------------
# 2️⃣ Check one frame (for RGB vs grayscale)
# -----------------------------
frame_path = "./frames/capture_200.jpg"
if os.path.exists(frame_path):
    img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    print(f"\n🖼 Frame shape: {img.shape}")
    if len(img.shape) == 3 and img.shape[2] == 3:
        print("🎨 Detected: RGB (3 channels)")
    else:
        print("⚫ Detected: Grayscale")
else:
    print(f"⚠️ Missing frame: {frame_path}")

# -----------------------------
# 3️⃣ Check GPU availability
# -----------------------------
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    name = torch.cuda.get_device_name(i)
    free, total = torch.cuda.mem_get_info()
    print(f"\n💻 GPU: {name}")
    print(f"💾 VRAM total (GB): {round(total/1024**3, 2)}")
    print(f"💨 VRAM free  (GB): {round(free/1024**3, 2)}")
else:
    print("\n🚫 CUDA not available")
