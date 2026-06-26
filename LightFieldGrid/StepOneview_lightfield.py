import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the 4D light field array
L = np.load("./results/light_field_array.npy")
U, V, H, W, C = L.shape
print("Loaded light field shape:", L.shape)

# Create a big grid to visualize (same as your original)
grid = np.zeros((U * H, V * W, C), dtype=np.uint8)

for u in range(U):
    for v in range(V):
        y0, y1 = u * H, (u + 1) * H
        x0, x1 = v * W, (v + 1) * W
        grid[y0:y1, x0:x1, :] = L[u, v]

# Save or view the grid
cv2.imwrite("./results/light_field_preview.png", grid)
print("✅ Saved preview as ./results/light_field_preview.png")

# Optional: show smaller version on screen
small = cv2.resize(grid, (W // 2, H // 2))
cv2.imshow("Light Field Views", small)
cv2.waitKey(0)
cv2.destroyAllWindows()
