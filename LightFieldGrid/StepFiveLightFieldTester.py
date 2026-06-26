import torch
import numpy as np
import os
import torch.nn.functional as F

# 🧠 Import your trained model class from your training file
from StepFive_Train3DObjectRecognition import LightField3DCNN


# ==========================
# Light Field Tester Class
# ==========================
class LightFieldTester:
    def __init__(self, model_path, data_root, class_names, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = LightField3DCNN(num_classes=len(class_names)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.class_names = class_names
        self.data_root = data_root
        print(f"✅ Model loaded from {model_path}")

    def preprocess_volume(self, vol_path):
        """Loads and preprocesses a single Light Field volume (same as training)"""
        volume = np.load(vol_path).astype(np.float32)
        volume = volume[:, ::2, ::4]
        volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-5)
        volume = np.expand_dims(volume, axis=0)
        tensor = torch.tensor(volume).unsqueeze(0).to(self.device)
        return tensor

    def predict_one(self, vol_path):
        """Predict the class for one sample"""
        x = self.preprocess_volume(vol_path)
        with torch.no_grad():
            out = self.model(x)
            probs = F.softmax(out, dim=1)[0].cpu().numpy()
            pred_idx = np.argmax(probs)
        pred_label = self.class_names[pred_idx]
        print(f"📦 File: {os.path.basename(vol_path)} → 🧩 Predicted: {pred_label} ({probs[pred_idx]*100:.2f}%)")
        return pred_label, probs[pred_idx]

    def test_folder(self):
        """Loop through all samples in your data folder"""
        print("🔍 Testing all samples from:", self.data_root)
        for cls in self.class_names:
            cls_path = os.path.join(self.data_root, cls)
            for sample_dir in os.listdir(cls_path):
                vol_path = os.path.join(cls_path, sample_dir, "lightfield_volume.npy")
                if os.path.exists(vol_path):
                    self.predict_one(vol_path)


# ==========================
# Run Testing
# ==========================
DATA_ROOT = "./results/data_lightfield"
MODEL_PATH = "./results/output_StepFive3DObjectRecognition/best_model.pt"

# same order as your training dataset folders
CLASS_NAMES = ["ball", "cup", "scissors"]

tester = LightFieldTester(MODEL_PATH, DATA_ROOT, CLASS_NAMES)

# 🧪 Test one specific sample
tester.predict_one("./results/data_lightfield/cup/sample_01/lightfield_volume.npy")

# 🧪 OR test all samples in your dataset
tester.test_folder()
