# Sixth_Train_dCNN_AugmentedModel.py
# Train a compact 3D CNN on EPI volumes (original + augmented)
# Dataset layout expected:
# /mnt/d/thesisProject/LightFieldGrid/levelone/results/volumes/train_ready/
#   ├── cup/
#   └── orange/

import os, glob, math, random
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------
# Config
# -----------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

ROOT = "/mnt/d/thesisProject/LightFieldGrid/levelone/results/volumes/train_ready"
CLASSES = ["orange", "cup"]
NUM_SLICES = 27
BATCH_SIZE = 4
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SAVE_PATH = os.path.join(ROOT, "EPI3DNet_best_augV2.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Utilities
# -----------------------
def list_class_files(root: str, cls: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, cls, "*.npy")))

def ensure_depth_27(v: np.ndarray, target: int = 27) -> np.ndarray:
    D = v.shape[0]
    if D == target:
        return v
    if D > target:
        start = (D - target) // 2
        return v[start:start + target]
    pad_front = (target - D) // 2
    pad_back = target - D - pad_front
    return np.pad(v, ((pad_front, pad_back), (0, 0), (0, 0)), mode="reflect")

def normalize_minmax(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin + 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v - vmin) / (vmax - vmin)

# -----------------------
# Dataset
# -----------------------
class EPIVolumeFolderDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        v = np.load(path)
        v = ensure_depth_27(v, NUM_SLICES)
        v = normalize_minmax(v)
        v = np.expand_dims(v, axis=0)
        t = torch.from_numpy(v).float()
        y = torch.tensor(label, dtype=torch.long)
        return t, y

# -----------------------
# Model (compact 3D CNN)
# -----------------------
class EPI3DNet(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(p=0.25)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x

# -----------------------
# Data Split (Train / Val / Test)
# -----------------------
def make_items(root: str, classes: List[str]) -> Tuple[List, List, List]:
    train_items, val_items, test_items = [], [], []
    for ci, cls in enumerate(classes):
        files = list_class_files(root, cls)
        # 70% train, 30% temp
        train_paths, temp_paths = train_test_split(files, test_size=0.3, random_state=SEED)
        # 15% val, 15% test
        val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=SEED)

        train_items += [(p, ci) for p in train_paths]
        val_items += [(p, ci) for p in val_paths]
        test_items += [(p, ci) for p in test_paths]

        print(f"Class '{cls}': total={len(files)} | train={len(train_paths)} | val={len(val_paths)} | test={len(test_paths)}")

    return train_items, val_items, test_items

# -----------------------
# Train / Eval Functions
# -----------------------
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def train_one_epoch(model, loader, optim, loss_fn) -> Tuple[float, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        b = y.size(0)
        total_loss += loss.item() * b
        total_acc += accuracy(logits.detach(), y) * b
        n += b
    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, loss_fn) -> Tuple[float, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = loss_fn(logits, y)
        b = y.size(0)
        total_loss += loss.item() * b
        total_acc += accuracy(logits, y) * b
        n += b
    return total_loss / n, total_acc / n

# -----------------------
# Main
# -----------------------
def main():
    print(f"Using device: {DEVICE}")
    train_items, val_items, test_items = make_items(ROOT, CLASSES)

    if len(train_items) == 0 or len(val_items) == 0:
        raise RuntimeError("No train/val items found. Check your train_ready folders.")

    # Data loaders
    train_ds = EPIVolumeFolderDataset(train_items)
    val_ds = EPIVolumeFolderDataset(val_items)
    test_ds = EPIVolumeFolderDataset(test_items)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # Model + Optimizer
    model = EPI3DNet(n_classes=len(CLASSES)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_acc = 0.0
    patience, num_bad = 15, 0

    # ---------------- Train ----------------
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, loss_fn)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn)
        scheduler.step(va_loss)

        print(f"[{epoch:03d}/{EPOCHS}] TrainLoss={tr_loss:.4f} Acc={tr_acc:.3f} | ValLoss={va_loss:.4f} Acc={va_acc:.3f}")

        if va_acc > best_val_acc + 1e-4:
            best_val_acc = va_acc
            torch.save({"model": model.state_dict(), "classes": CLASSES}, SAVE_PATH)
            print(f"Saved new best model (val_acc={best_val_acc:.3f})")
            num_bad = 0
        else:
            num_bad += 1
            if num_bad >= patience:
                print("Early stopping (no val improvement).")
                break

    # ---------------- Test ----------------
    print("\n Evaluating best model on unseen test data...")
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"Final Test Accuracy = {test_acc:.3f} | Test Loss = {test_loss:.4f}")

    print("Training + Testing completed successfully.")

if __name__ == "__main__":
    main()
