''' k-fold on: batch size, learning rate, weight decay, betas, use_scheduler
link results:
https://docs.google.com/document/d/1ySgvGH9z321AE_rY8TMAAjj2yOaVZU7DVQKm3NU0qVM/edit?tab=t.0


import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from torchmetrics import Accuracy

# ----- Parameters -----
k_folds = 5
batch_sizes = [32, 64]
learning_rates = [7.5e-4, 1e-3, 2.5e-3, 5e-3]
use_scheduler_opts = [False, True]
weight_decays = [0.0, 1e-4, 1e-3]
betas_opts = [(0.9, 0.999), (0.9, 0.98)]
num_epochs = 90
fs = 128  # sampling frequency
segment_length = 2 * fs
step_train = int(0.5 * segment_length)
step_test = segment_length
num_class = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Data Loading -----
folder_path = "D:/Student_Projects/Thesis Gioele/Codes/Data/00/00_"
end_path = "_full_filtered.npy"
relax = np.load(folder_path + "relax" + end_path, allow_pickle=True)
stroop = np.load(folder_path + "stroop" + end_path, allow_pickle=True)

# ----- Big segmentation -----
num_segments = 15
big_chunks, big_labels = [], []
for sub in relax:
    for i in range(num_segments):
        big_chunks.append(sub[i*(len(sub)//num_segments):(i+1)*(len(sub)//num_segments), :])
        big_labels.append(0)
for sub in stroop:
    for i in range(num_segments):
        big_chunks.append(sub[i*(len(sub)//num_segments):(i+1)*(len(sub)//num_segments), :])
        big_labels.append(1)
big_chunks = np.stack(big_chunks)
big_labels = np.array(big_labels)

def segment_windows(data, labels, seg_len, step):
    segs, labs = [], []
    for trial, lab in zip(data, labels):
        for start in range(0, trial.shape[0] - seg_len + 1, step):
            segs.append(trial[start:start+seg_len, :].T)
            labs.append(lab)
    return np.stack(segs), np.array(labs)

all_segs, all_labs = segment_windows(big_chunks, big_labels, segment_length, step_train)

# Precompute chunk index mapping
chunk_idxs = []
idx = 0
for _ in range(len(big_chunks)):
    count = (big_chunks.shape[1] - segment_length) // step_train + 1
    chunk_idxs.append(list(range(idx, idx+count)))
    idx += count

# ----- Model Definition -----
class EEGNet(nn.Module):
    def __init__(self, num_class=2):  
        super(EEGNet, self).__init__()
        
        # Layer 1
        self.conv2d = nn.Conv2d(1, 16, kernel_size=(1, 64), padding=0)
        self.Batch_normalization_1 = nn.BatchNorm2d(16)
        self.Elu = nn.ELU()
        self.Dropout = nn.Dropout(0.25)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.Depthwise_conv2D = nn.Conv2d(16, 4, kernel_size=(2, 32))
        self.Batch_normalization_2 = nn.BatchNorm2d(4)
        self.Average_pooling2D_1 = nn.AvgPool2d(kernel_size=(2, 4))
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.Separable_conv2D_depth = nn.Conv2d(4, 4, kernel_size=(8, 4), padding=0, groups=4)
        self.Separable_conv2D_point = nn.Conv2d(4, 4, kernel_size=(1, 1))
        self.Batch_normalization_3 = nn.BatchNorm2d(4)
        self.Average_pooling2D_2 = nn.AvgPool2d(kernel_size=(2, 4))
        
        # Layer 4
        self.Flatten = nn.Flatten()
        self.Dense = nn.Linear(384, num_class) # 384 for 2 seconds segment
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Layer 1
        x = self.conv2d(x)
        x = self.Batch_normalization_1(x)
        x = self.Elu(x)
        x = self.Dropout(x)
        
        # Layer 2
        x = self.padding1(x)
        x = self.Depthwise_conv2D(x)
        x = self.Batch_normalization_2(x)
        x = self.Elu(x)
        x = self.Dropout(x)
        x = self.Average_pooling2D_1(x)
        
        # Layer 3
        x = self.padding2(x)
        x = self.Separable_conv2D_depth(x)
        x = self.Separable_conv2D_point(x)
        x = self.Batch_normalization_3(x)
        x = self.Elu(x)
        x = self.Dropout(x)
        x = self.Average_pooling2D_2(x)
        
        # Layer 4
        x = self.Flatten(x)
        x = self.Dense(x)
        x = self.Softmax(x)
        return x

# ----- Experiment Loop -----
results = []
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for bs in batch_sizes:
    for lr in learning_rates:
        for wd in weight_decays:
            for betas in betas_opts:
                for use_scheduler in use_scheduler_opts:
                    fold_accs = []
                    print(f"Config: BS={bs}, LR={lr}, WD={wd}, Betas={betas}, Scheduler={use_scheduler}")
                    for fold, (train_chunks, test_chunks) in enumerate(kf.split(big_chunks), 1):
                        # build indices
                        train_idx = [i for fc in train_chunks for i in chunk_idxs[fc]]
                        test_idx  = [i for fc in test_chunks  for i in chunk_idxs[fc]]
                        X_train = torch.tensor(all_segs[train_idx], dtype=torch.float32).unsqueeze(1).to(device)
                        y_train = torch.tensor(all_labs[train_idx], dtype=torch.long).to(device)
                        X_test  = torch.tensor(all_segs[test_idx], dtype=torch.float32).unsqueeze(1).to(device)
                        y_test  = torch.tensor(all_labs[test_idx], dtype=torch.long).to(device)

                        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True)
                        test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=bs, shuffle=False)

                        model = EEGNet(num_class).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
                        if use_scheduler:
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
                        loss_fn = nn.CrossEntropyLoss()

                        # train
                        for epoch in range(1, num_epochs+1):
                            model.train()
                            for inputs, labs in train_loader:
                                outputs = model(inputs)
                                loss = loss_fn(outputs, labs)
                                optimizer.zero_grad(); loss.backward(); optimizer.step()
                            if use_scheduler:
                                scheduler.step()

                        # eval
                        model.eval()
                        acc_metric = Accuracy(task="multiclass", num_classes=num_class).to(device)
                        with torch.no_grad():
                            for inputs, labs in test_loader:
                                preds = model(inputs)
                                acc_metric(preds, labs)
                        acc = acc_metric.compute().item()
                        print(f"  Fold {fold} accuracy: {acc*100:.2f}%")
                        fold_accs.append(acc)

                    avg_acc = np.mean(fold_accs)
                    results.append({
                        'batch_size': bs,
                        'lr': lr,
                        'weight_decay': wd,
                        'betas': betas,
                        'use_scheduler': use_scheduler,
                        'fold_accuracies': fold_accs,
                        'avg_test_acc': avg_acc
                    })
                    print(f"-> Avg acc: {avg_acc*100:.2f}%\n")

# ----- Save Results -----
df = pd.DataFrame(results)
df = df.sort_values('avg_test_acc', ascending=False).reset_index(drop=True)
print("\n=== BEST CONFIG ===")
print(df.iloc[0])
df.to_csv("kfold_results.csv", index=False)
print("\nResults saved to kfold_results.csv")
'''


# k-fold on the segment_length
# results at the link: https://docs.google.com/spreadsheets/d/16KXpBSxOAq1D5y0xtPjN3F-HD_jYSBubfq-v4VNJY-g/edit?usp=sharing
import os
import sys
import glob
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fs = 128  # sampling frequency
batch_size = 64
num_class = 2
num_folds = 5
segment_factors = [1, 2, 3, 4, 5] # signal segment lengths in seconds
num_epochs = 80

# Data paths
folder_path = "D:/Student_Projects/Thesis Gioele/Codes/Data/00/00_"
end_path = "_zNorm_full_filtered.npy"
relax_data = np.load(folder_path + "relax" + end_path, allow_pickle=True)
stroop_data = np.load(folder_path + "stroop" + end_path, allow_pickle=True)

# Prepare trials by splitting each subject into folds
relax_trials = [seg for sub in relax_data for seg in np.array_split(sub, num_folds)]
stroop_trials = [seg for sub in stroop_data for seg in np.array_split(sub, num_folds)]
all_trials = np.array(relax_trials + stroop_trials)
all_labels = np.array([0]*len(relax_trials) + [1]*len(stroop_trials))

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42) # divide into 5 folds

# EEGNet definition
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


# segment trials into windows
def segment_trials(trials, labels, segment_length, step):
    segs, labs = [], []
    for trial, lab in zip(trials, labels):
        for start in range(0, trial.shape[0] - segment_length + 1, step):
            segs.append(trial[start:start+segment_length, :].T)
            labs.append(lab)
    return np.stack(segs), np.array(labs)

# Main k-fold over segment lengths
results = []
for factor in segment_factors: # factor is the segment length in seconds
    seg_len = factor * fs
    step_train = int(0.5 * seg_len)
    step_test = seg_len
    fold_idx = 0
    for train_idx, test_idx in kf.split(all_trials):
        fold_idx += 1
        X_train_trials = all_trials[train_idx]
        y_train_trials = all_labels[train_idx]
        X_test_trials = all_trials[test_idx]
        y_test_trials = all_labels[test_idx]

        X_train, y_train = segment_trials(X_train_trials, y_train_trials, seg_len, step_train)
        X_test, y_test = segment_trials(X_test_trials, y_test_trials, seg_len, step_test)

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = EEGNet(num_channels=all_trials.shape[2], segment_length=seg_len).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4, betas=(0.9,0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        loss_fn = nn.CrossEntropyLoss().to(device)

        # Training loop
        for epoch in range(1, num_epochs+1):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step(); optimizer.zero_grad()
            scheduler.step()

        # Evaluation
        model.eval()
        acc_metric = Accuracy(task="multiclass", num_classes=num_class).to(device)
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                acc_metric(outputs, targets)
        test_acc = acc_metric.compute().item()
        results.append({"factor_s": factor, "fold": fold_idx, "accuracy": test_acc})
        print("Factor:", factor, "Fold:", fold_idx, "Accuracy:", test_acc)

# Save CSV
import pandas as pd
out_df = pd.DataFrame(results)
out_path = "eegnet_kfold_results.csv"
out_df.to_csv(out_path, index=False)
print(f"Results saved to {out_path}")
