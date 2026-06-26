import numpy as np
import open3d as o3d
import os

# -------------- CONFIG --------------
base_dir   = "/mnt/d/thesisProject/mini_kitti_dataset/sequences/00"
npy_dir    = os.path.join(base_dir, "converted_npy")
calib_cam2cam = os.path.join(base_dir, "calib_cam_to_cam.txt")
calib_velo2cam = os.path.join(base_dir, "calib_velo_to_cam.txt")

center_frame = 0       # central frame index (change if needed)
window_size  = 2       # frames: t-2..t+2
output_ply   = f"fused_refined_cloud_{center_frame:06d}.ply"


# -------------- CALIBRATION LOADER --------------
def read_calib_file(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                try:
                    data[key] = np.array([float(x) for x in val.split()])
                except ValueError:
                    # skip non-numeric lines like calib_time
                    pass
    return data

cam = read_calib_file(calib_cam2cam)
velo = read_calib_file(calib_velo2cam)

P2 = cam["P_rect_02"].reshape(3, 4)
R  = velo["R"].reshape(3, 3)
T  = velo["T"].reshape(3,)   # 3-vector


# -------------- STEP 1: MULTI-FRAME LIDAR FUSION --------------
all_pts_cam = []   # fused geometry

for idx in range(center_frame - window_size, center_frame + window_size + 1):
    if idx < 0:
        continue
    lidar_file = os.path.join(npy_dir, f"frame_{idx:06d}_lidar.npy")
    if not os.path.exists(lidar_file):
        print(f"⚠️ Skipping missing LiDAR frame: {lidar_file}")
        continue

    lidar = np.load(lidar_file)[:, :3]  # XYZ only

    # Transform LiDAR → camera for this frame
    pts_cam = (R @ lidar.T + T.reshape(3,1)).T   # (N,3)
    all_pts_cam.append(pts_cam)

if len(all_pts_cam) == 0:
    raise RuntimeError("No LiDAR frames loaded, check paths!")

fused_pts_cam = np.vstack(all_pts_cam)   # (N_total, 3)
N = fused_pts_cam.shape[0]
print("✅ Fused LiDAR points:", N)


# -------------- STEP 2: SLIDING WINDOW COLOR REFINEMENT --------------
# We will project fused points onto multiple RGB frames
# from center_frame-window_size ... center_frame+window_size

final_colors = np.zeros((N, 3), dtype=float)
count        = np.zeros(N, dtype=float)

# We assume each frame has corresponding RGB .npy: frame_xxx_img.npy
for idx in range(center_frame - window_size, center_frame + window_size + 1):
    if idx < 0:
        continue
    img_file = os.path.join(npy_dir, f"frame_{idx:06d}_img.npy")
    if not os.path.exists(img_file):
        print(f"⚠️ Skipping missing image frame: {img_file}")
        continue

    img = np.load(img_file)   # H×W×3
    H, W = img.shape[:2]

    # homogeneous coordinates for projection
    pts_hom = np.hstack((fused_pts_cam, np.ones((N,1))))
    proj = (P2 @ pts_hom.T).T  # (N,3)
    proj[:,0] /= proj[:,2]
    proj[:,1] /= proj[:,2]

    # valid mask (inside image + in front of camera)
    m = (proj[:,2] > 0) & \
        (proj[:,0] >= 0) & (proj[:,0] < W) & \
        (proj[:,1] >= 0) & (proj[:,1] < H)

    u = proj[m,0].astype(int)
    v = proj[m,1].astype(int)

    final_colors[m] += img[v, u] / 255.0
    count[m] += 1

# avoid division by zero
valid = count > 0
final_colors[valid] /= count[valid][:, None]

# for points never seen in any image, keep them black (0,0,0) or set gray
final_colors[~valid] = 0.0

print("✅ Color refinement done. Colored points:", valid.sum())


# -------------- STEP 3: SAVE FUSED + REFINED POINT CLOUD --------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(fused_pts_cam)
pcd.colors = o3d.utility.Vector3dVector(final_colors)

o3d.io.write_point_cloud(output_ply, pcd)
print("💾 Saved fused & refined cloud →", output_ply)
