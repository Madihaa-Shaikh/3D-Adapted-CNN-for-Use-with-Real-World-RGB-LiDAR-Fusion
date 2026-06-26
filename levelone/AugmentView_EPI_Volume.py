import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
# Example path tum apna file choose karo
path = "/mnt/d/thesisProject/LightFieldGrid/levelone/results/volumes/augmented/orange/orange_aug4.npy"

# 🔹 load the 3D light field volume
V = np.load(path)
print("✅ Loaded volume:", V.shape)

# 🔹 loop through depth slices (like scanning through the cube)
for i in range(V.shape[0]):
    frame = V[i]              # slice at depth i
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    frame = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_TURBO)

    cv2.putText(frame, f"Slice {i+1}/{V.shape[0]}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("EPI Volume Viewer", frame)

    key = cv2.waitKey(150)  # wait 150ms → control speed
    if key == 27:           # press ESC to exit early
        break

cv2.destroyAllWindows()
