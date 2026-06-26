import cv2
import numpy as np
import os

# Folder containing your EPIs
epi_folder = "./results/epi"
os.makedirs(epi_folder, exist_ok=True)

# List of EPI files you want to enhance
epi_files = ["horizontal_epi.png", "vertical_epi.png"]

for file in epi_files:
    path = os.path.join(epi_folder, file)
    img = cv2.imread(path)

    if img is None:
        print(f"⚠️ Could not open {file}")
        continue

    print(f"🎨 Enhancing {file} ...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1️⃣ Histogram Equalization (for contrast)
    eq = cv2.equalizeHist(gray)
    cv2.imwrite(os.path.join(epi_folder, f"eq_{file}"), eq)

    # 2️⃣ CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    cv2.imwrite(os.path.join(epi_folder, f"clahe_{file}"), clahe_img)

    # 3️⃣ Sobel Edge Detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(epi_folder, f"sobel_{file}"), sobel_norm)

    # 4️⃣ Colormap Visualization (Jet)
    color_jet = cv2.applyColorMap(eq, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(epi_folder, f"colormap_jet_{file}"), color_jet)

    # 5️⃣ Colormap Visualization (Plasma)
    color_plasma = cv2.applyColorMap(eq, cv2.COLORMAP_PLASMA)
    cv2.imwrite(os.path.join(epi_folder, f"colormap_plasma_{file}"), color_plasma)

    # 6️⃣ Edge Overlay (Combine edges with color)
    edges = cv2.Canny(gray, 50, 150)
    overlay = cv2.addWeighted(color_jet, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.5, 0)
    cv2.imwrite(os.path.join(epi_folder, f"edge_overlay_{file}"), overlay)

print("✅ All enhanced EPI visualizations saved in ./results/epi")
