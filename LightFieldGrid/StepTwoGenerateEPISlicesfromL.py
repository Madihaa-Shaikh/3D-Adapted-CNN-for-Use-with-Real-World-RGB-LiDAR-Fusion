import numpy as np
import cv2
import os

# Load 4D Light Field array
L = np.load("./results/light_field_array.npy")  # shape: (U,V,H,W,3)
U, V, H, W, C = L.shape
os.makedirs("./results/epi", exist_ok=True)

# Choose how many slices to extract
num_slices = 10
y_indices = np.linspace(H//4, (3*H)//4, num_slices, dtype=int)
x_indices = np.linspace(W//4, (3*W)//4, num_slices, dtype=int)

epi_list = []

for i, y in enumerate(y_indices):
    v = V // 2
    epi = L[:, v, y, :, :]
    gray = cv2.cvtColor(epi, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"./results/epi/output_StepTwoGenerateEPISlicesFromL/h_epi_{i:02d}.png", gray)
    epi_list.append(gray)

for j, x in enumerate(x_indices):
    u = U // 2
    epi = L[u, :, :, x, :]
    gray = cv2.cvtColor(epi, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"./results/epi/output_StepTwoGenerateEPISlicesFromL/v_epi_{j:02d}.png", gray)
    epi_list.append(gray)

print(f"✅ Generated {len(epi_list)} EPIs (horizontal + vertical) → ./results/epi/output_StepTwoGenerateEPISlicesFromL/")
