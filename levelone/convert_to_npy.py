import os
import numpy as np
import cv2
import glob

dataset_dir = "/mnt/d/thesisProject/mini_kitti_dataset/sequences/00"

image_dir = os.path.join(dataset_dir, "image_0", "data")
lidar_dir = os.path.join(dataset_dir, "velodyne")
output_dir = os.path.join(dataset_dir, "converted_npy")

os.makedirs(output_dir, exist_ok=True)

image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.bin")))

print(f"Found {len(image_files)} images and {len(lidar_files)} LiDAR files")

for idx, (img_path, lidar_path) in enumerate(zip(image_files, lidar_files)):
    img = cv2.imread(img_path)
    lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    
    np.save(os.path.join(output_dir, f"frame_{idx:06d}_img.npy"), img)
    np.save(os.path.join(output_dir, f"frame_{idx:06d}_lidar.npy"), lidar_points)

print(f"\n✅ Conversion complete! Saved all .npy files to:\n{output_dir}")
