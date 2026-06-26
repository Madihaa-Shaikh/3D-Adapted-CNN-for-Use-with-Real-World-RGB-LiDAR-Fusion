# 🚗 PointCloud-to-Voxel-3DCNN

> **A Deep Learning Pipeline for Voxelizing LiDAR Point Clouds and Classifying 3D Objects using a Custom 3D Convolutional Neural Network (3D CNN).**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Dataset](https://img.shields.io/badge/Dataset-KITTI-green)
![Task](https://img.shields.io/badge/Task-3D%20Object%20Classification-orange)

---

# 📖 Overview

This repository presents a complete end-to-end pipeline for **3D object classification** using LiDAR point cloud data from the **KITTI Vision Benchmark Dataset**.

The proposed framework converts raw LiDAR point clouds into **voxel grid representations**, enabling a custom **3D Convolutional Neural Network (3D CNN)** to automatically learn spatial and geometric features for object classification.

Unlike traditional machine learning approaches that rely on handcrafted features, the proposed model performs **automatic feature learning** directly from voxelized point clouds.

This work was developed as part of a **Master's Thesis in Artificial Intelligence** at **BTU Cottbus–Senftenberg, Germany**.

---

# 🎯 Research Objective

The objective of this research is to investigate the effectiveness of voxel-based deep learning for LiDAR point cloud classification.

The proposed framework focuses on:

- Converting raw LiDAR point clouds into voxel grids.
- Learning spatial and geometric features automatically.
- Classifying objects into:
  - 🚗 Car
  - 🚶 Pedestrian
  - 🚴 Cyclist
- Evaluating the performance of a lightweight 3D CNN architecture.

---

# 🔬 Methodology

The complete workflow consists of the following stages.

```text
Raw LiDAR Point Cloud
          │
          ▼
Camera Calibration
          │
          ▼
3D Bounding Box Cropping
          │
          ▼
Point Cloud Preprocessing
          │
          ▼
Voxelization
          │
          ▼
Binary Occupancy Grid
          │
          ▼
3D CNN
          │
          ▼
Classification
```

---

# 📂 Repository Structure

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
│   └── Additional Experiments
│
├── LightFieldGrid/
├── .gitignore
└── README.md
```

---

# 🧠 Deep Learning Architecture

The proposed network consists of:

- 3D Convolution Layers
- Batch Normalization
- ReLU Activation
- Max Pooling
- Fully Connected Layers
- Dropout
- Softmax Classification

---

# ⚙️ Technologies Used

- Python
- PyTorch
- NumPy
- Open3D
- OpenCV
- Matplotlib
- KITTI Dataset

---

# 📊 Training Configuration

| Parameter | Value |
|-----------|--------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Cross Entropy Loss |
| Batch Size | 16 |
| Activation Function | ReLU |
| Normalization | BatchNorm3D |
| Dropout | 0.3 |

---

# 📈 Results

The proposed framework successfully classified voxelized LiDAR objects from the KITTI dataset.

### Object Classes

- 🚗 Car
- 🚶 Pedestrian
- 🚴 Cyclist

The lightweight 3D CNN demonstrated effective learning of spatial and geometric information while maintaining computational efficiency.

---

# ⭐ Key Contributions

- End-to-end LiDAR preprocessing pipeline.
- Camera calibration and coordinate transformation.
- Object extraction using KITTI annotations.
- Binary occupancy voxelization.
- Lightweight custom 3D CNN.
- Automatic feature learning.
- Multi-class object classification.

---

# 🚀 Future Work

Future improvements include:

- PointNet
- PointNet++
- Point Transformer
- Sparse Convolution Networks
- SECOND
- VoxelNet
- Voxel R-CNN
- Real-time 3D Object Detection

---

# 💾 Installation

Clone the repository

```bash
git clone https://github.com/Madihaa-Shaikh/PointCloud-to-Voxel-3DCNN.git
```

Move into the project directory

```bash
cd PointCloud-to-Voxel-3DCNN
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run training

```bash
python levelone/KittiDataset/scripts/4_train_3dcnn.py
```

---

# 📚 Dataset

This project uses the **KITTI Vision Benchmark Suite**.

Official Website:

https://www.cvlibs.net/datasets/kitti/

---

# 👩‍💻 Author

**Madiha Shaikh**

Master's in Artificial Intelligence

BTU Cottbus–Senftenberg

Germany

GitHub:

https://github.com/Madihaa-Shaikh

---

# 📄 License

This repository is intended for academic and research purposes.
