# ==============================================================
# StepSix_ModelEvaluationAndVisualization.py
# Evaluate trained 3D Light-Field CNN, plot metrics & confusion matrix
# ==============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ==============================================================
# 1️⃣ Configuration
# ==============================================================
DATA_ROOT = "./results/data_lightfield"
MODEL_PATH = "./results/output_StepFive3DObjectRecognition/best_model.pt"
OUTPUT_DIR = "./results/output_StepSixEvaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
print(f"💻 Using device: {DEVICE}")

# ==============================================================
# 2️⃣ Import same Dataset + Model class used in training
# ==============================================================
from StepFive_Train3DObjectRecognition import LightFieldDataset, LightField3DCNN

# ==============================================================
# 3️⃣ Load dataset (validation set)
# ==============================================================
from sklearn.model_selection import train_test_split

full_dataset = LightFieldDataset(DATA_ROOT)
_, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42, shuffle=True)
val_data = torch.utils.data.Subset(full_dataset, val_idx)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

class_names = full_dataset.class_names
print(f"✅ Loaded {len(val_data)} validation samples from classes: {class_names}")

# ==============================================================
# 4️⃣ Load trained model
# ==============================================================
model = LightField3DCNN(num_classes=len(class_names)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"📦 Model loaded from: {MODEL_PATH}")

# ==============================================================
# 5️⃣ Evaluation Loop
# ==============================================================
y_true, y_pred = [], []
criterion = nn.CrossEntropyLoss()
val_loss = 0.0

with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        val_loss += loss.item()
        preds = out.argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.cpu().numpy())

val_loss /= len(val_loader)

print("\n📊 Evaluation Results:")
print("Average Validation Loss:", round(val_loss, 4))
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

# ==============================================================
# 6️⃣ Confusion Matrix
# ==============================================================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (3D Light-Field CNN)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()
print(f"✅ Confusion matrix saved → {OUTPUT_DIR}/confusion_matrix.png")

# ==============================================================
# 7️⃣ Accuracy & Loss Curves (if log file exists)
# ==============================================================
# If you saved train/val history during training, visualize it here
log_file = "./results/output_StepFive3DObjectRecognition/training_log.npy"
if os.path.exists(log_file):
    history = np.load(log_file, allow_pickle=True).item()
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')

    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))
    plt.close()
    print(f"✅ Training curves saved → {OUTPUT_DIR}/training_curves.png")
else:
    print("⚠️ No training log found; skipping loss/accuracy plot.")

# ==============================================================
# 8️⃣ Example Predictions
# ==============================================================
plt.figure(figsize=(8,4))
for i, (x, y) in enumerate(val_loader):
    if i >= 3: break  # show 3 examples
    x = x.to(DEVICE)
    pred = model(x).argmax(1).item()
    true_label = class_names[y.item()]
    pred_label = class_names[pred]

    vol = x.cpu().numpy()[0,0,:,:,:]
    slice_img = vol[vol.shape[0]//2, :, :]  # middle slice

    plt.subplot(1,3,i+1)
    plt.imshow(slice_img, cmap='jet')
    plt.title(f"T:{true_label}\nP:{pred_label}")
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "example_predictions.png"))
plt.close()
print(f"✅ Example prediction slices saved → {OUTPUT_DIR}/example_predictions.png")

print("\n🎉 Evaluation complete! All results saved in:", OUTPUT_DIR)
