# ThirdTrain3D_CNN_HeberStyle.py
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, glob, os

DATA_ROOT = "/mnt/d/thesisProject/LightFieldGrid/levelone/results/volumes"
CLASSES   = ["orange", "cup"]
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, LR, BATCH = 150, 1e-3, 4

class LF3DDataset(Dataset):
    def __init__(self, root, classes):
        self.paths, self.labels = [], []
        for idx, cls in enumerate(classes):
            for f in glob.glob(os.path.join(root, cls, "*.npy")):
                self.paths.append(f)
                self.labels.append(idx)
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        vol = np.load(self.paths[idx]).astype(np.float32)
        vol = np.expand_dims(vol, 0)    # (1,D,H,W)
        return torch.tensor(vol), torch.tensor(self.labels[idx])

# ------------------ Heber-style 3D CNN ------------------
class EPI3DNet(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm3d(32), nn.BatchNorm3d(64), nn.BatchNorm3d(128)
        self.pool = nn.MaxPool3d(2)
        self.drop = nn.Dropout3d(0.3)
        self.fc1  = nn.Linear(128 * 3 * 16 * 16, 256)  # adjust based on your input
        self.fc2  = nn.Linear(256, ncls)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------ Training Loop ------------------
def train_one_epoch(model, loader, opt, loss_fn):
    model.train(); tot_loss=acc=0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward(); opt.step()
        tot_loss += loss.item()*x.size(0)
        acc += (out.argmax(1)==y).sum().item()
    return tot_loss/len(loader.dataset), acc/len(loader.dataset)

def eval_one_epoch(model, loader, loss_fn):
    model.eval(); tot_loss=acc=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            tot_loss += loss_fn(out,y).item()*x.size(0)
            acc += (out.argmax(1)==y).sum().item()
    return tot_loss/len(loader.dataset), acc/len(loader.dataset)

def main():
    ds = LF3DDataset(DATA_ROOT, CLASSES)
    n = len(ds)
    ntrain = int(0.8*n)
    train,val = torch.utils.data.random_split(ds, [ntrain, n-ntrain])
    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH)

    model = EPI3DNet(len(CLASSES)).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print("💻 Using:", DEVICE)
    for e in range(1,EPOCHS+1):
        tl,ta = train_one_epoch(model, train_loader, opt, loss_fn)
        vl,va = eval_one_epoch(model, val_loader, loss_fn)
        print(f"[{e:03d}/{EPOCHS}] TrainLoss={tl:.4f} Acc={ta:.3f} | ValLoss={vl:.4f} Acc={va:.3f}")

    torch.save(model.state_dict(), os.path.join(DATA_ROOT, "EPI3DNet_best.pth"))
    print("✅ Model saved.")

if __name__=="__main__":
    main()
