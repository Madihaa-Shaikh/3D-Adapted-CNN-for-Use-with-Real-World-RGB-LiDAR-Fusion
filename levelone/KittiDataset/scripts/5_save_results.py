import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt

# ============================
# CONFIG (same as training)
# ============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

VOXEL_ROOT = Path("/mnt/d/thesisProject/LightFieldGrid/levelone/KittiDataset/Kitti/voxels")
MODEL_PATH = Path("/mnt/d/thesisProject/LightFieldGrid/levelone/KittiDataset/models/voxel3d_best.pth")

# Results base folder (new run output will go inside)
RESULTS_BASE = Path("/mnt/d/thesisProject/LightFieldGrid/levelone/KittiDataset/results")

CLASSES = ["car", "pedestrian", "cyclist"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

BATCH_SIZE = 16
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# DATASET
# ============================
class KittiVoxelDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        vox = data["voxels"].astype(np.float32)  # (32,32,32)
        vox = np.expand_dims(vox, axis=0)        # (1,32,32,32)
        x = torch.from_numpy(vox)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def load_all_voxel_paths():
    file_paths, labels = [], []
    for cls in CLASSES:
        cls_dir = VOXEL_ROOT / cls
        files = sorted(cls_dir.glob("*.npz"))
        print(f"📂 Class '{cls}' -> {len(files)} samples")
        for f in files:
            file_paths.append(f)
            labels.append(CLASS_TO_IDX[cls])
    return file_paths, labels

# ============================
# MODEL (same as training)
# ============================
class Voxel3DNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # -> (B,16,16,16,16)

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # -> (B,32,8,8,8)

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # -> (B,64,4,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================
# SAVE HELPERS
# ============================
def save_confusion_matrix_png(cm, classes, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Greys",
        linewidths=0.5,
        linecolor="black",
        cbar=True,
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={"size": 11}
    )

    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.title("Normalized Confusion Matrix", fontsize=13)

    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ============================
# SAVE CLASS DISTRIBUTION BAR CHART
# ============================
def save_class_distribution_bar_chart(labels, classes, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    # Display names for graph
    display_classes = ['Car', 'Pedestrian', 'Cyclist']

    # Count samples per class
    counts = [labels.count(i) for i in range(len(classes))]

    plt.figure(figsize=(6,4.5))

    bars = plt.bar(display_classes, counts)

    # Add values on top of bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            str(count),
            ha='center',
            fontsize=10
        )

    plt.xlabel("Object Classes", fontsize=11)
    plt.ylabel("Number of Samples", fontsize=11)
    plt.title("Class Distribution in the Voxelized Dataset", fontsize=13)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ============================
# MAIN
# ============================
def main():
    print("🖥 Using device:", DEVICE)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Create a NEW results folder each time (timestamp)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_BASE / f"eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset + split (same seed for reproducibility)
    file_paths, labels = load_all_voxel_paths()
    dataset = KittiVoxelDataset(file_paths, labels)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load model weights (READ ONLY - does not change file)
    model = Voxel3DNet(num_classes=len(CLASSES)).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Evaluate
    y_true, y_pred = [], []
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    cm = confusion_matrix(y_true, y_pred)

    cm_normalized = cm.astype("float") / np.maximum(
        cm.sum(axis=1, keepdims=True), 1
    )

    save_confusion_matrix_png(
        cm_normalized,
        CLASSES,
        out_dir / "normalized_confusion_matrix.png"
    )
    save_class_distribution_bar_chart(
        labels,
        CLASSES,
        out_dir / "class_distribution.png"
    )
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)

    # Save outputs
    (out_dir / "classification_report.txt").write_text(report)
    
    metrics = {
        "model_path": str(MODEL_PATH),
        "voxel_root": str(VOXEL_ROOT),
        "seed": SEED,
        "val_split": VAL_SPLIT,
        "batch_size": BATCH_SIZE,
        "validation_accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": cm_normalized.tolist(),
        "classes": CLASSES,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n✅ EVALUATION DONE (model unchanged).")
    print(f"✅ Validation Accuracy: {acc:.4f}")
    print(f"📁 Saved results to: {out_dir}")
    print(f"🖼 normalized_confusion_matrix.png")
    print(f"📝 classification_report.txt")
    print(f"🔢 class_distribution.png\n")
    print(f"🔢 metrics.json\n")

if __name__ == "__main__":
    main()
