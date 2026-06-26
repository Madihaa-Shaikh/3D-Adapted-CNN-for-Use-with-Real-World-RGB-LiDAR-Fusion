# PointCloud-to-Voxel-3DCNN

> A deep learning pipeline for voxelizing LiDAR point clouds and classifying 3D objects using a custom 3D CNN on the KITTI dataset.

---

## Overview

This project presents an end-to-end pipeline for **3D object classification** using LiDAR point cloud data. Raw point clouds are processed, converted into voxel grids, and classified using a custom 3D Convolutional Neural Network (3D CNN).

The project was developed as part of a Master's research in Artificial Intelligence.

---

## Features

- LiDAR Point Cloud Processing
- Point Cloud Preprocessing
- Point Cloud Cropping
- Point Cloud Voxelization
- Binary Occupancy Encoding
- 3D CNN Architecture
- Data Augmentation
- Model Training and Evaluation
- 3D Object Classification

---

## Dataset

**KITTI Vision Benchmark Suite**

Classes:

- Car
- Pedestrian
- Cyclist

Dataset Link:

https://www.cvlibs.net/datasets/kitti/

---

## Project Pipeline

```text
Raw LiDAR Point Cloud
          │
          ▼
Point Cloud Preprocessing
          │
          ▼
Object Cropping
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

## Repository Structure

```text
PointCloud-to-Voxel-3DCNN
│
├── LightFieldGrid/
│   ├── Training Scripts
│   ├── Voxelization
│   ├── Reconstruction
│   └── Evaluation
│
├── levelone/
│   ├── Data Preparation
│   ├── Augmentation
│   ├── Visualization
│   ├── Training
│   └── Utilities
│
├── .gitignore
├── LightFieldGrid.sln
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
|-----------|--------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Cross Entropy Loss |
| Batch Size | 16 |
| Activation | ReLU |
| Normalization | BatchNorm3D |

---

## Results

The proposed 3D CNN successfully classifies voxelized LiDAR objects from the KITTI dataset.

The model demonstrates strong classification performance while maintaining a lightweight architecture suitable for efficient training.

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

Clone the repository

```bash
git clone https://github.com/Madihaa-Shaikh/PointCloud-to-Voxel-3DCNN.git
```

Go to project directory

```bash
cd PointCloud-to-Voxel-3DCNN
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run training

```bash
python StepFive_Train3DObjectRecognition.py
```

---

## Author

**Madiha Shaikh**

Master's in Artificial Intelligence

BTU Cottbus–Senftenberg

Germany

GitHub:

https://github.com/Madihaa-Shaikh

---

## License

This repository is intended for academic and research purposes.
