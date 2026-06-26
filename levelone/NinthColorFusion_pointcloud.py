import numpy as np
import open3d as o3d
import os

# ---------- Paths ----------
lidar_path = "/mnt/d/thesisProject/mini_kitti_dataset/sequences/00/converted_npy/frame_000000_lidar.npy"
image_path = "/mnt/d/thesisProject/mini_kitti_dataset/sequences/00/converted_npy/frame_000000_img.npy"
calib_velo2cam = "/mnt/d/thesisProject/mini_kitti_dataset/sequences/00/calib_velo_to_cam.txt"
calib_cam2cam = "/mnt/d/thesisProject/mini_kitti_dataset/sequences/00/calib_cam_to_cam.txt"
output_ply = "/mnt/d/thesisProject/LightFieldGrid/levelone/colored_pointcloud.ply"

# ---------- Load data ----------
if not os.path.exists(lidar_path):
    raise FileNotFoundError(f"Lidar file not found: {lidar_path}")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

lidar = np.load(lidar_path)[:, :3]
image = np.load(image_path)

# ---------- Helper: Parse calibration ----------
def read_calib_file(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            try:
                data[key.strip()] = np.array([float(x) for x in value.strip().split()])
            except ValueError:
                print(f"Skipping non-numeric line: {key}")
                continue
    return data

# ---------- Load calibration ----------
velo2cam = read_calib_file(calib_velo2cam)
cam2cam = read_calib_file(calib_cam2cam)

# Extract transformation matrices safely
R = velo2cam.get("R", np.eye(3)).reshape(3, 3)
T = velo2cam.get("T", np.zeros(3)).reshape(3, 1)

# Rectification + Projection (fallbacks if missing)
if "R_rect_00" in cam2cam:
    R_rect = cam2cam["R_rect_00"].reshape(3, 3)
else:
    print("⚠️ R_rect_00 not found in calib_cam_to_cam.txt — using identity matrix.")
    R_rect = np.eye(3)

if "P2" in cam2cam:
    P2 = cam2cam["P2"].reshape(3, 4)
else:
    found = [v for v in cam2cam.values() if v.size == 12]
    if len(found) > 0:
        P2 = found[0].reshape(3, 4)
        print("⚠️ Using fallback projection matrix.")
    else:
        raise ValueError("❌ No projection matrix found in calib_cam_to_cam.txt")

# Combine velo->cam transform
Tr_velo_to_cam = np.hstack((R, T))

# ---------- Transform points ----------
N = lidar.shape[0]
pts_hom = np.hstack((lidar, np.ones((N, 1))))

# Transform to camera coordinates
pts_cam = (R_rect @ (Tr_velo_to_cam @ pts_hom.T)).T

# Project to image plane
proj = (P2 @ np.hstack((pts_cam, np.ones((N, 1)))).T).T
proj[:, 0] /= proj[:, 2]
proj[:, 1] /= proj[:, 2]

# ---------- Select visible points ----------
H, W = image.shape[:2]
mask = (
    (proj[:, 0] >= 0) & (proj[:, 0] < W) &
    (proj[:, 1] >= 0) & (proj[:, 1] < H) &
    (proj[:, 2] > 0)
)

proj = proj[mask]
pts_cam = pts_cam[mask]

# ---------- Get colors ----------
colors = image[proj[:, 1].astype(np.int32), proj[:, 0].astype(np.int32)] / 255.0

# ---------- Save colored PLY ----------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_cam)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(output_ply, pcd)
print("✅ Colored point cloud saved at:", output_ply)
