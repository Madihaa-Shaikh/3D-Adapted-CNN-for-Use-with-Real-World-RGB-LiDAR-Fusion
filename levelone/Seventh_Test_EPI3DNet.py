import torch, numpy as np
from Sixth_Train_dCNN_AugmentedModel import EPI3DNet, normalize_minmax, ensure_depth_27

# Path to saved model and test file
MODEL_PATH = "/mnt/d/thesisProject/LightFieldGrid/levelone/results/volumes/train_ready/EPI3DNet_best_augV2.pth"
TEST_FILE  = "/mnt/d/thesisProject/LightFieldGrid/levelone/results/volumes/train_ready/orange/orange_aug3.npy"

# Load checkpoint
ckpt = torch.load(MODEL_PATH, map_location="cpu")
classes = ckpt["classes"]
model = EPI3DNet(n_classes=len(classes))
model.load_state_dict(ckpt["model"])
model.eval()

# Load and preprocess volume
V = np.load(TEST_FILE)
V = ensure_depth_27(V)
V = normalize_minmax(V)
V = np.expand_dims(V, (0,1))     # (B,C,D,H,W)
x = torch.from_numpy(V).float()

# Predict
with torch.no_grad():
    out = model(x)
    pred = out.argmax(dim=1).item()
    print(f"✅ Prediction: {classes[pred]} (class index={pred})")
