import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv
import seaborn as sns
# ============================
# REPRODUCIBILITY (SEED)
# ============================

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================
# CONFIG
# ============================

VOXEL_ROOT = Path("D:/thesisProject/LightFieldGrid/levelone/KittiDataset/Kitti/voxels")
MODEL_DIR  = Path("D:/thesisProject/LightFieldGrid/levelone/KittiDataset/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["car", "pedestrian", "cyclist"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

VOXEL_SIZE = 32           # 32x32x32
BATCH_SIZE = 16
EPOCHS     = 20           # You can increase later
LR         = 1e-3
VAL_SPLIT  = 0.2          # 80% train, 20% val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 Using device: {DEVICE}")


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
        path = self.file_paths[idx]
        label = self.labels[idx]

        data = np.load(path)
        vox = data["voxels"]  # shape: (32,32,32), dtype uint8

        # Convert to float32 and add channel dimension: (1, D, H, W)
        vox = vox.astype(np.float32)
        vox = np.expand_dims(vox, axis=0)

        x = torch.from_numpy(vox)          # (1, 32, 32, 32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y


def load_all_voxel_paths():
    file_paths = []
    labels = []

    for cls in CLASSES:
        cls_dir = VOXEL_ROOT / cls
        cls_files = sorted(cls_dir.glob("*.npz"))

        print(f"📂 Class '{cls}' -> {len(cls_files)} samples")

        for f in cls_files:
            file_paths.append(f)
            labels.append(CLASS_TO_IDX[cls])

    return file_paths, labels


# ============================
# 3D CNN MODEL
# ============================

class Voxel3DNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            # Input: (B, 1, 32, 32, 32)
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),        # -> (B,16,16,16,16)

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),        # -> (B,32,8,8,8)

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),        # -> (B,64,4,4,4)
        )

        # 64 * 4 * 4 * 4 = 4096
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


# ============================
# TRAIN / EVAL FUNCTIONS
# ============================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            running_loss += loss.item() * xb.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


# ============================
# MAIN
# ============================


def evaluate_full(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    return all_labels, all_preds



def main():
    # Load file paths + labels
    file_paths, labels = load_all_voxel_paths()

    dataset = KittiVoxelDataset(file_paths, labels)
    dataset_size = len(dataset)
    val_size = int(dataset_size * VAL_SPLIT)
    train_size = dataset_size - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"\n📊 Dataset split: {train_size} train, {val_size} val")

    # using GPU
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model, loss, optimizer
    model = Voxel3DNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_model_path = MODEL_DIR / "voxel3d_best.pth"

    # History lists
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("\n🚀 Starting training...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"[Epoch {epoch:03d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Saved new best model with Val Acc={best_val_acc:.3f} at {best_model_path}")

    print("\n✅ Training finished.")
    print(f"🏆 Best Val Acc: {best_val_acc:.3f}")

    epochs = range(1, EPOCHS + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "loss_curve.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "accuracy_curve.png")
    plt.show()

    # ===== FINAL EVALUATION =====
    labels, preds = evaluate_full(model, val_loader, DEVICE)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    print("\n📊 Final Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Save TXT
    with open(MODEL_DIR / "metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")

    # Save CSV
    
    with open(MODEL_DIR / "metrics.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", acc])
        writer.writerow(["Precision", precision])
        writer.writerow(["Recall", recall])
        writer.writerow(["F1-score", f1])

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    np.savetxt(MODEL_DIR / "confusion_matrix.txt", cm, fmt='%d')

   

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES,
                yticklabels=CLASSES,
                cbar=True,
                linewidths=0.5,
                linecolor='gray')

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=300)
    plt.show()

    import pandas as pd

    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Value": [acc, precision, recall, f1]
    }

    df = pd.DataFrame(metrics_data)

    plt.figure(figsize=(5, 2))
    plt.axis('off')

    table = plt.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    plt.title("Model Performance Metrics", fontsize=14)

    plt.savefig(MODEL_DIR / "metrics_table.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
