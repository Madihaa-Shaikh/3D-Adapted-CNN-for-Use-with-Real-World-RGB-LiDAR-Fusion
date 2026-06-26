import os
import numpy as np
import random
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================
# Configuration
# ==========================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATA_ROOT = "./results/data_lightfield"  # Dataset folder
BATCH_SIZE = 4
EPOCHS = 150
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("💻 Using device:", DEVICE)

# ==========================
# Dataset class
# ==========================
class LightFieldDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []
        self.class_names = sorted(os.listdir(data_root))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_path = os.path.join(data_root, cls)
            for sample_dir in os.listdir(cls_path):
                sample_path = os.path.join(cls_path, sample_dir)
                if not os.path.isdir(sample_path):
                    continue
                # Load all .npy volumes
                for file in os.listdir(sample_path):
                    if file.endswith(".npy"):
                        vol_path = os.path.join(sample_path, file)
                        self.samples.append((vol_path, self.class_to_idx[cls]))

        print(f"✅ Found {len(self.samples)} samples across {len(self.class_names)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol_path, label = self.samples[idx]
        volume = np.load(vol_path).astype(np.float32)

        # 🔹 Downsample
        volume = volume[:, ::2, ::4]

        # 🔹 Random augmentations
        # if random.random() > 0.5:
        #     volume = np.flip(volume, axis=-1).copy()
        # if random.random() > 0.5:
        #     volume = np.flip(volume, axis=-2).copy()
        if random.random() > 0.5:
            angle = random.choice([-10, -5, 5, 10])
            volume = rotate(volume, angle, axes=(-2, -1), reshape=False)
        # if random.random() > 0.5:
        #     factor = 1.0 + (0.1 * np.random.randn())
        #     volume *= factor
        # if random.random() > 0.5:
        #     noise = np.random.normal(0, 0.01, size=volume.shape)
        #     volume += noise

        # 🔹 Normalize and clip
        volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-5)
        volume = np.clip(volume, -3, 3)
        volume = np.expand_dims(volume, axis=0)
        return torch.tensor(volume), torch.tensor(label)

# ==========================
# Model definition
# ==========================
class LightField3DCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 16, (3, 1, 7), stride=(1, 1, 3), padding=(1, 0, 3))
        self.norm1 = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, (3, 1, 5), stride=(1, 1, 2), padding=(1, 0, 2))
        self.norm2 = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, (3, 1, 3), stride=(1, 1, 2), padding=(1, 0, 1))
        self.norm3 = nn.BatchNorm3d(64)

        self.conv4 = nn.Conv3d(64, 64, (1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.norm4 = nn.BatchNorm3d(64)

        self.conv5 = nn.Conv3d(128, 64, (3, 3, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.norm5 = nn.BatchNorm3d(64)

        # Pool smaller to prevent flatten explosion
        self.pool = nn.AdaptiveAvgPool3d((2, 1, 16))
        self.fc1 = nn.Linear(64 * 2 * 1 * 16, 128)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        for conv, norm in [
            (self.conv1, self.norm1),
            (self.conv2, self.norm2),
            (self.conv3, self.norm3),
            (self.conv4, self.norm4),
        ]:
            x = F.relu(norm(conv(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# ==========================
# Load data
# ==========================
dataset = LightFieldDataset(DATA_ROOT)
train_idx, val_idx = train_test_split(
    range(len(dataset)), test_size=0.2, random_state=42, shuffle=True
)
train_data = torch.utils.data.Subset(dataset, train_idx)
val_data = torch.utils.data.Subset(dataset, val_idx)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Loaded {len(train_data)} training and {len(val_data)} validation samples")
print("Classes:", dataset.class_names)

# ==========================
# Initialize model
# ==========================
model = LightField3DCNN(num_classes=len(dataset.class_names)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
criterion = nn.CrossEntropyLoss()

print(f"✅ Model initialized. Starting training with LR={LR} and Batch Size={BATCH_SIZE}")

# ==========================
# Training loop
# ==========================
best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    train_loss, correct = 0.0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()

    train_acc = correct / len(train_data)
    scheduler.step()

    # Validation
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()
            val_correct += (out.argmax(1) == y).sum().item()

    val_acc = val_correct / len(val_data)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            model.state_dict(),
            "./results/output_StepFive3DObjectRecognition/best_model.pt",
        )

    print(
        f"[Epoch {epoch+1:03}/{EPOCHS}] "
        f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.3f}"
    )

print(
    f"💾 Training complete! Best validation accuracy: {best_val_acc*100:.2f}%"
)
