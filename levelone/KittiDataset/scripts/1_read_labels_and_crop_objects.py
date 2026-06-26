import os
import numpy as np
from pathlib import Path

ROOT = Path("/mnt/d/thesisProject/LightFieldGrid/levelone/KittiDataset/Kitti")

# Correct KITTI paths
label_dir    = ROOT / "data_object_label_2" / "training" / "label_2"
image_dir    = ROOT / "data_object_image_2" / "training" / "image_2"
calib_dir    = ROOT / "data_object_calib" / "training" / "calib"
velodyne_dir = ROOT / "data_object_velodyne" / "training" / "velodyne"

PROCESSED = ROOT / "processed"

# Pre-create class folders
for cls in ["car", "cyclist", "pedestrian"]:
    (PROCESSED / cls).mkdir(parents=True, exist_ok=True)


# ==============================
# LOAD CALIBRATION
# ==============================
def load_calib(frame_id):
    calib_file = calib_dir / f"{frame_id}.txt"
    data = {}

    if not calib_file.exists():
        print(f"⚠️ Calibration missing for frame {frame_id}")
        return None

    with open(calib_file, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            data[key] = np.array([float(x) for x in val.split()])

    try:
        P2        = data["P2"].reshape(3, 4)
        R0_rect   = data["R0_rect"].reshape(3, 3)
        Tr_velo   = data["Tr_velo_to_cam"].reshape(3, 4)
    except KeyError:
        print(f"⚠️ Missing parameters in calib file: {calib_file}")
        return None

    # make 4x4 matrices
    Tr_velo_4x4 = np.eye(4)
    Tr_velo_4x4[:3, :] = Tr_velo

    R0_4x4 = np.eye(4)
    R0_4x4[:3, :3] = R0_rect

    # Full transform: X_cam = R0_rect * Tr_velo_to_cam * X_velo
    T_velo_to_cam = R0_4x4 @ Tr_velo_4x4

    return {
        "P2": P2,
        "R0_4x4": R0_4x4,
        "Tr_velo_4x4": Tr_velo_4x4,
        "T_velo_to_cam": T_velo_to_cam
    }


# ==============================
# LOAD LIDAR (VELODYNE)
# ==============================
def load_velodyne(path):
    pc = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pc[:, :3]   # keep x, y, z only


# ==============================
# TRANSFORM VELODYNE → CAMERA
# ==============================
def velo_to_camera(pc_velo, T_velo_to_cam):
    """
    pc_velo: (N,3) in Velodyne frame
    returns: (N,3) in rectified camera-2 frame
    """
    # add homogeneous coordinate
    N = pc_velo.shape[0]
    pc_h = np.hstack([pc_velo, np.ones((N, 1))])   # (N,4)

    # row-vector form: X_cam = X_velo * T^T
    pc_cam_h = pc_h @ T_velo_to_cam.T              # (N,4)
    return pc_cam_h[:, :3]


# ==============================
# 3D CROPPING IN CAMERA FRAME
# ==============================
def crop_lidar(pc_cam, box):
    # KITTI: h, w, l, x, y, z, ry
    # (x, y, z) is bottom-center of box in camera coords
    h, w, l, x, y, z, ry = box

    # convert bottom-center → geometric center
    y_center = y - h / 2.0
    center = np.array([x, y_center, z])

    # rotation around Y-axis (KITTI convention)
    R = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [ 0,         1, 0        ],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # move points into box local frame
    pc_local = pc_cam - center     # translate
    pc_local = pc_local @ R.T      # rotate

    # inside-box mask (x=l, y=h, z=w in local frame)
    mask = (
        (pc_local[:, 0] > -l/2) & (pc_local[:, 0] < l/2) &
        (pc_local[:, 1] > -h/2) & (pc_local[:, 1] < h/2) &
        (pc_local[:, 2] > -w/2) & (pc_local[:, 2] < w/2)
    )

    return pc_cam[mask]


# ==============================
# MAIN LOOP
# ==============================
valid_classes = ["Car", "Cyclist", "Pedestrian"]

label_list = sorted(os.listdir(label_dir))
print("Found labels:", len(label_list))

for f in label_list:

    frame_id = f.split(".")[0]

     # LIMIT frames (only process first 300 frames)
    if int(frame_id) > 300:
        print("Frame limit reached — stopping early at frame:", frame_id)
        break

    calib = load_calib(frame_id)
    if calib is None:
        continue

    T_velo_to_cam = calib["T_velo_to_cam"]

    velo_path = velodyne_dir / f"{frame_id}.bin"
    if not velo_path.exists():
        print(f"⚠️ LiDAR missing for frame {frame_id}")
        continue

    # 1) Load LiDAR in Velodyne frame
    pc_velo = load_velodyne(velo_path)

    # 2) Transform to camera frame
    pc_cam = velo_to_camera(pc_velo, T_velo_to_cam)

    # Read all object labels for this frame
    with open(label_dir / f, 'r') as ftxt:
        lines = ftxt.readlines()

    for i, line in enumerate(lines):
        parts = line.split()
        cls = parts[0]

        # Skip non-required classes
        if cls not in valid_classes:
            continue

        # Extract 3D box info (KITTI format)
        h = float(parts[8])
        w = float(parts[9])
        l = float(parts[10])
        x = float(parts[11])
        y = float(parts[12])
        z = float(parts[13])
        ry = float(parts[14])

        box = (h, w, l, x, y, z, ry)

        # Crop LiDAR points inside this 3D box
        cropped = crop_lidar(pc_cam, box)

        # Allow even small objects but avoid completely empty
        if len(cropped) == 0:
            print(f"⚠️ No lidar points in {cls} box, frame {frame_id}, obj {i}")
            continue

        save_dir = PROCESSED / cls.lower()
        save_path = save_dir / f"{frame_id}_{i}_{cls.lower()}.npy"

        np.save(save_path, cropped)
        print(f"Saved {cls} points:", save_path)

print("\n Done! Saved all CAR, CYCLIST, PEDESTRIAN point clouds.\n")
