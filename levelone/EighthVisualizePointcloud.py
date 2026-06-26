import numpy as np
import open3d as o3d
import os

# Path jahan npy files hain
data_dir = "/mnt/d/thesisProject/mini_kitti_dataset/sequences/00/converted_npy"

# Example: ek frame load karte hain
img = np.load(os.path.join(data_dir, "frame_000000_img.npy"))
lidar = np.load(os.path.join(data_dir, "frame_000000_lidar.npy"))

# LiDAR points ko Open3D format me convert karo
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(lidar[:, :3])
o3d.io.write_point_cloud("output_pointcloud.ply", pcd)
print("✅ Point cloud saved as output_pointcloud.ply")

# Simple visualize

