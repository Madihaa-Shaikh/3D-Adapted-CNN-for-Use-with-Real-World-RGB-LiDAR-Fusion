import numpy as np
import open3d as o3d
from pathlib import Path

# CHANGE THIS TO THE FOLDER YOU WANT TO VISUALIZE
pc_folder = Path("/mnt/d/thesisProject/LightFieldGrid/levelone/KittiDataset/processed/car")

# List all crops
pc_files = sorted(list(pc_folder.glob("*.npy")))

print(f"Found {len(pc_files)} point-cloud crops")

def visualize_npy(path):
    pc = np.load(path)

    # Create point cloud for Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # Color (optional): make cars red, pedestrians green, cyclists blue
    if "car" in str(path):
        color = [1, 0, 0]
    elif "pedestrian" in str(path):
        color = [0, 1, 0]
    elif "cyclist" in str(path):
        color = [0, 0, 1]
    else:
        color = [1, 1, 1]

    pcd.paint_uniform_color(color)

    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name=str(path))


# View the first 10 crops
for file in pc_files[:10]:
    print("Visualizing:", file)
    visualize_npy(file)
