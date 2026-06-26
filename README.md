# PointCloud-to-Voxel-3DCNN

A deep learning pipeline for voxelizing LiDAR point clouds and classifying 3D objects using a custom 3D CNN on the KITTI dataset.

---

## Project Pipeline

```text
Raw LiDAR Point Cloud
        ↓
Point Cloud Preprocessing
        ↓
Object Cropping
        ↓
Voxelization
        ↓
Binary Occupancy Grid
        ↓
3D CNN
        ↓
Classification
```

---

## Repository Structure

```text
PointCloud-to-Voxel-3DCNN
│
├── levelone/
│   ├── KittiDataset/
│   │   ├── scripts/
│   │   │   ├── 1_read_labels_and_crop_objects.py
│   │   │   ├── 2_visualize_pointcloud.py
│   │   │   ├── 3_voxelize.py
│   │   │   ├── 4_train_3dcnn.py
│   │   │   └── 5_save_results.py
│   │   │
│   │   ├── models/
│   │   └── results/
│   │
│   └── additional_experiments/
│
├── LightFieldGrid/
├── .gitignore
└── README.md
```

---

## Deep Learning Model

The proposed network consists of:

- 3D Convolution Layers
- Batch Normalization
- ReLU Activation
- Max Pooling
- Fully Connected Layers
- Softmax Classification

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Open3D
- OpenCV
- Matplotlib
- KITTI Dataset

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Cross Entropy Loss |
| Batch Size | 16 |
| Activation | ReLU |
| Normalization | BatchNorm3D |

---

## Results

The proposed 3D CNN classifies voxelized LiDAR objects from the KITTI dataset into:

- Car
- Pedestrian
- Cyclist

---

## Future Improvements

- Sparse Convolution Networks
- PointNet++
- Point Transformer
- Voxel R-CNN
- SECOND
- Real-time 3D Object Detection

---

## Installation

```bash
git clone https://github.com/Madihaa-Shaikh/PointCloud-to-Voxel-3DCNN.git
cd PointCloud-to-Voxel-3DCNN
pip install -r requirements.txt
```

---

## Author

**Madiha Shaikh**  
Master's in Artificial Intelligence  
BTU Cottbus-Senftenberg, Germany
