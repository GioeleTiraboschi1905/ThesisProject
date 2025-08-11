import os
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from torchmetrics import Accuracy

# ----- Configuration -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fs = 128  # sampling frequency (Hz)
segment_length = 2 * fs  # 2 seconds
step_train = int(0.5 * segment_length)
step_test  = segment_length
k_folds = 5
num_epochs = 90
num_class = 2

batch_sizes      = [32, 64]
learning_rates   = [7.5e-4, 1e-3, 2.5e-3, 5e-3]
use_scheduler_opts = [False, True]
weight_decays    = [0.0, 1e-4, 1e-3]
betas_opts       = [(0.9, 0.999), (0.8, 0.98)]

# ----- Data Loading -----
folder_path = "D:/Student_Projects/Thesis Gioele/Codes/Data/00/00_"
end_path    = "_full_filtered.npy"
relax = np.load(os.path.join(folder_path, "relax"  + end_path), allow_pickle=True)
stroop= np.load(os.path.join(folder_path, "stroop" + end_path), allow_pickle=True)

# Big segmentation into 15 chunks per trial
num_segments = 15
big_chunks, big_labels = [], []
for sub in relax:
    for i in range(num_segments):
        chunk = sub[i*(len(sub)//num_segments):(i+1)*(len(sub)//num_segments), :]
        big_chunks.append(chunk)
        big_labels.append(0)
for sub in stroop:
    for i in range(num_segments):
        chunk = sub[i*(len(sub)//num_segments):(i+1)*(len(sub)//num_segments), :]
        big_chunks.append(chunk)
        big_labels.append(1)
big_chunks = np.stack(big_chunks)      # shape: (N_trials, T, channels)
big_labels = np.array(big_labels)      # shape: (N_trials,)

def segment_windows(data, labels, seg_len, step):
    X, y = [], []
    for trial, lab in zip(data, labels):
        for start in range(0, trial.shape[0] - seg_len + 1, step):
            # transpose so shape = (channels, seg_len)
            X.append(trial[start:start+seg_len, :].T)
            y.append(lab)
    return np.stack(X), np.array(y)

all_segs, all_labs = segment_windows(big_chunks, big_labels, segment_length, step_train)

# Precompute trial→window indices
chunk_idxs = []
idx = 0
windows_per_trial = (big_chunks.shape[1] - segment_length)//step_train + 1
for _ in range(len(big_chunks)):
    chunk_idxs.append(list(range(idx, idx + windows_per_trial)))
    idx += windows_per_trial

# ----- Model Definition -----
class EEGNet(nn.Module):
    def __init__(self, num_channels, segment_length, num_class=2):
        super().__init__()

        self.num_channels = num_channels
        self.segment_length = segment_length

        # Layer 1
        self.conv2d = nn.Conv2d(1, 16, kernel_size=(1, 64), padding=0)
        self.bn1    = nn.BatchNorm2d(16)
        self.elu    = nn.ELU()
        self.drop   = nn.Dropout(0.25)

        # Layer 2
        self.pad1       = nn.ZeroPad2d((16, 17, 0, 1))
        self.depth_conv = nn.Conv2d(16, 4, kernel_size=(2, 32))
        self.bn2        = nn.BatchNorm2d(4)
        self.pool1      = nn.AvgPool2d((2, 4))

        # Layer 3
        self.pad2       = nn.ZeroPad2d((2, 1, 4, 3))
        self.sep_dep    = nn.Conv2d(4, 4, kernel_size=(8, 4), groups=4)
        self.sep_point  = nn.Conv2d(4, 4, kernel_size=1)
        self.bn3        = nn.BatchNorm2d(4)
        self.pool2      = nn.AvgPool2d((2, 4))

        # use a dummy pass to infer flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, segment_length)
            x = self._forward_features(dummy)
            n_features = x.shape[1]

        # final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, num_class),
            nn.Softmax(dim=1)
        )

    def _forward_features(self, x):
        x = self.conv2d(x)
        x = self.bn1(x);  x = self.elu(x);  x = self.drop(x)

        x = self.pad1(x)
        x = self.depth_conv(x)
        x = self.bn2(x);  x = self.elu(x);  x = self.drop(x)
        x = self.pool1(x)

        x = self.pad2(x)
        x = self.sep_dep(x)
        x = self.sep_point(x)
        x = self.bn3(x);  x = self.elu(x);  x = self.drop(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.classifier(x)
        return x

# ----- Experiment Loop -----
results = []
kf = KFold(n_splits=k_folds, shuffle=True)

for bs in batch_sizes:
    for lr in learning_rates:
        for wd in weight_decays:
            for betas in betas_opts:
                for use_scheduler in use_scheduler_opts:
                    fold_accs = []
                    for fold, (train_ch, test_ch) in enumerate(kf.split(big_chunks), 1):
                        # map trials → windows
                        train_idx = [i for tc in train_ch for i in chunk_idxs[tc]]
                        test_idx  = [i for tc in test_ch  for i in chunk_idxs[tc]]
                        X_train = torch.tensor(all_segs[train_idx], dtype=torch.float32).unsqueeze(1).to(device)
                        y_train = torch.tensor(all_labs[train_idx], dtype=torch.long).to(device)
                        X_test  = torch.tensor(all_segs[test_idx],  dtype=torch.float32).unsqueeze(1).to(device)
                        y_test  = torch.tensor(all_labs[test_idx],  dtype=torch.long).to(device)

                        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True)
                        test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=bs, shuffle=False)

                        model = EEGNet(num_class).to(device)
                        optimizer = optim.Adam(model.parameters(),
                                               lr=lr,
                                               betas=betas,
                                               weight_decay=wd)
                        if use_scheduler:
                            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
                        loss_fn = nn.CrossEntropyLoss()

                        # Train
                        for epoch in range(num_epochs):
                            model.train()
                            for xb, yb in train_loader:
                                pred = model(xb)
                                loss = loss_fn(pred, yb)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                            if use_scheduler:
                                scheduler.step()

                        # Eval
                        model.eval()
                        accm = Accuracy(task="multiclass", num_classes=num_class).to(device)
                        with torch.no_grad():
                            for xb, yb in test_loader:
                                pred = model(xb)
                                accm(pred, yb)
                        acc = accm.compute().item()
                        fold_accs.append(acc)

                    avg_acc = np.mean(fold_accs)
                    results.append({
                        'batch_size': bs,
                        'lr': lr,
                        'weight_decay': wd,
                        'betas': betas,
                        'use_scheduler': use_scheduler,
                        'fold_accs': fold_accs,
                        'avg_acc': avg_acc
                    })
                    print(f"BS={bs}, LR={lr}, WD={wd}, Betas={betas}, Scheduler={use_scheduler} -> Avg Acc: {avg_acc:.4f}")

# Save results
df = pd.DataFrame(results)
df = df.sort_values('avg_acc', ascending=False).reset_index(drop=True)
print("\nBest config:\n", df.iloc[0])
df.to_csv("kfold_hyperparam_sweep_results.csv", index=False)
print("Results saved to kfold_hyperparam_sweep_results.csv")
