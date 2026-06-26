import cv2
import numpy as np
import os, glob

# ---------- Orange detection HSV range ----------
# (H:0-179, S:0-255, V:0-255)
COLOR_RANGES = {
    "orange": ((5, 100, 50), (30, 255, 255))   # tuned for bright + dark orange
}

# ---------- Input / Output directories ----------
INPUT_DIR  = "/mnt/d/thesisProject/LightFieldGrid/levelone/frames_raw"
OUTPUT_DIR = "/mnt/d/thesisProject/LightFieldGrid/levelone/frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)
for cls in COLOR_RANGES.keys():
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

print(f"📂 INPUT_DIR: {INPUT_DIR}")
print(f"🖼️ Found {len(glob.glob(os.path.join(INPUT_DIR, '*.jpg')))} raw frames\n")

# ---------- Extractor function ----------
def extract_and_save(image_path, preview=False):
    img = cv2.imread(image_path)
    if img is None:
        print("⚠️ Could not read:", image_path)
        return

    # Enhance contrast for dark frames
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Small blur to reduce highlights
    img = cv2.GaussianBlur(img, (5, 5), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    base = os.path.basename(image_path)
    name = os.path.splitext(base)[0]

    lower, upper = COLOR_RANGES["orange"]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))


    b_channel = hsv[..., 0]
    mask[b_channel > 110] = 0

    # Morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ No orange detected in {name}")
        return

    # Choose largest circular contour
    best_cnt = None
    best_score = 0
    ih, iw = img.shape[:2]
    cx, cy = iw//2, ih//2

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.55:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        center_dist = np.sqrt((bx + bw/2 - cx)**2 + (by + bh/2 - cy)**2)
        score = circularity * (1.0 / (1 + center_dist / cx))
        if score > best_score:
            best_cnt = cnt
            best_score = score

    if best_cnt is None:
        print(f"⚠️ No valid contour in {name}")
        return

    x, y, w, h = cv2.boundingRect(best_cnt)
    pad = 15
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(iw - x, w + 2*pad)
    h = min(ih - y, h + 2*pad)

    crop = img[y:y+h, x:x+w]
    out_path = os.path.join(OUTPUT_DIR, "orange", f"{name}_orange.png")
    cv2.imwrite(out_path, crop)
    print(f"✅ Saved → {out_path}")

    if preview:
        cv2.imshow("crop", crop)
        cv2.waitKey(100)
        cv2.destroyAllWindows()

# ---------- Main ----------
def main():
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
                   glob.glob(os.path.join(INPUT_DIR, "*.png")))
    if not paths:
        print("❌ No input images found!")
        return

    print(f"🔍 Found {len(paths)} image files for processing...")
    for p in paths:
        extract_and_save(p, preview=False)

    print("\n🎯 All orange crops completed successfully!")

if __name__ == "__main__":
    main()
