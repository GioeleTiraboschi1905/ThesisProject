# classifies EEG data. 
# v1: takes the data, segments, and then splits in train, val, test
# v2: splits the data in tr,val,test, at the beginning, and then segments the trials
# v2.1: segments the trials in smaller segments, then splits in train, val, test, and in the end segments the small trials in smaller segments

# %%
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# %% LOADING DATA
plot_results = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_path = "D:\\Student_Projects\\Thesis Gioele\\Codes\\Data\\01\\01_"
end_path = "_ica_v0.npy"

relax_data = np.load(folder_path + "relax" + end_path, allow_pickle=True)
stroop_data = np.load(folder_path + "stroop" + end_path, allow_pickle=True)

channels_32 = True 
if channels_32: num_channels = relax_data[0].shape[1]
else: # to train the model for testing on the OpenBCI data, which has 16 channels
    num_channels = 16
    idx = [i - 1 for i in [3, 32, 5, 30, 2, 7, 28, 1, 13, 22, 17, 16, 19, 14, 25, 10]] # channels similar to the channels in the OpenBCI device 
    relax_data = [sub[:, idx] for sub in relax_data]
    stroop_data = [sub[:, idx] for sub in stroop_data]
    
# %% data segmentation and division in train, val, test set.
# Divide each trial into 15 segments of 5 seconds each (6400 samples). Then split these 5-second blocks into train, val, test sets.
relax_data_divided = []
stroop_data_divided = []
num_segments = 15
len_segment = len(relax_data[0]) // num_segments # 5 seconds = 6400 samples

for sub in relax_data: # for each sub-> 9600x32
    for i in range(num_segments):
        relax_data_divided.append(sub[i*len_segment:(i+1)*len_segment, :] )

for sub in stroop_data:
    for i in range(num_segments):
        stroop_data_divided.append(sub[i*len_segment:(i+1)*len_segment, :])

tot_divided = np.stack(relax_data_divided + stroop_data_divided, axis=0)
all_labels  = np.array([0]*len(relax_data_divided) + [1]*len(stroop_data_divided))

# divide in construction and test set
trials_train_val, trials_test, labels_train_val, labels_test = train_test_split(
    tot_divided, all_labels,
    test_size=0.15, random_state=42, 
    stratify=all_labels
)

fs = 128  # sampling frequency
batch_size = 32
print("Batch size:", batch_size)
num_class = 2

def segment_data_trials(data, trial_labels, segment_length, step):
    segments, labels = [], []
    for trial, lab in zip(data, trial_labels):
        for start in range(0, trial.shape[0] - segment_length + 1, step):
            seg = trial[start:start+segment_length, :].T  # (channels, time)
            segments.append(seg)
            labels.append(lab)
    return np.stack(segments), np.array(labels)


factor = 2
segment_length = factor * fs
print("Segment length:", factor, "seconds")
step_train_val = int(0.5 * segment_length) # overlap = 1 - step_%
step_test      = int(1 * segment_length) # putting overlap in the test will only generate more samples for the test set, but not more information
print("Train Overlap:", 1 - (step_train_val / segment_length))

segments_tv, labs_tv = segment_data_trials(trials_train_val, labels_train_val, segment_length, step_train_val)
segments_test, labs_test = segment_data_trials(trials_test,       labels_test, segment_length, step_test)

# 3) Shuffle-and-split train+val segments into train and val sets
seg_train, seg_val, lab_train, lab_val = train_test_split(
    segments_tv, labs_tv,
    test_size=0.15, random_state=42,
    stratify=labs_tv
)

# 4) Convert to torch tensors and create DataLoaders;  add channel dimension with unsqueeze(1)
signal_train = torch.tensor(seg_train, dtype=torch.float32).unsqueeze(1)
signal_val   = torch.tensor(seg_val,   dtype=torch.float32).unsqueeze(1)
signal_test  = torch.tensor(segments_test, dtype=torch.float32).unsqueeze(1)

label_train = torch.tensor(lab_train, dtype=torch.long)
label_val   = torch.tensor(lab_val,   dtype=torch.long)
label_test  = torch.tensor(labs_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(signal_train, label_train), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(signal_val,   label_val),   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(signal_test,  label_test),  batch_size=batch_size, shuffle=False)

# EEGNet Model inspired from https://github.com/aliasvishnu/EEGNet/blob/master/EEGNet-PyTorch.ipynb
  
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

model = EEGNet(num_channels=num_channels, segment_length=segment_length).to(device)

# Functions for train and val
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    loss_train = AverageMeter()
    acc_train = Accuracy(task="multiclass", num_classes= num_class).to(device)
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        loss_train.update(loss.item())
        acc_train(outputs, targets.int())
        
    return model, loss_train.avg, acc_train.compute().item()

def validate(model, val_loader, loss_fn):
    model.eval()
    loss_val = AverageMeter()
    acc_val = Accuracy(task="multiclass", num_classes=num_class).to(device)
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            loss_val.update(loss.item())
            acc_val(outputs, targets.int())
    
    return loss_val.avg, acc_val.compute().item()

loss_train = []
acc_train = []
loss_val = []
acc_val = []

num_epochs = 101
tot_epochs = num_epochs
loss_fn= nn.CrossEntropyLoss().to(device)  
optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4, betas=(0.9,0.999)) 
use_scheduler = True
if use_scheduler: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

best_val_acc = 0.0
if channels_32: model_path = "D:\\Student_Projects\\Thesis Gioele\\Codes//Data//04//04_model_2cl_EEGNet.pt"
else: model_path = "D:\\Student_Projects\\Thesis Gioele\\Codes//Data//04//04_model_2cl_EEGNet_16.pt" # 16 channels

patience = 20  # Max epochs without val_acc improvement before early stopping
counter = 0  # Counts the epochs without any improvement

start_train = time.time()
for epoch in range(num_epochs):
    model, loss_train_ep, acc_train_ep = train_one_epoch(model, train_loader, loss_fn, optimizer)
  
    loss_train.append(loss_train_ep)
    acc_train.append(acc_train_ep)

    loss_val_ep, acc_val_ep = validate(model, val_loader, loss_fn)
    loss_val.append(loss_val_ep)
    acc_val.append(acc_val_ep)

    if acc_val_ep > best_val_acc:
        best_val_acc = acc_val_ep
        torch.save(model.state_dict(), model_path)
        counter = 0
    else:
        counter += 1 

    if counter >= patience:
        print(f"Early stopping ({patience} epochs) triggered after {epoch+1} epochs.")
        tot_epochs = epoch + 1
        break

    if (epoch % 10 == 5) or (epoch % 10 == 0): 
        print(f'epoch {epoch}:')
        print(f' Loss = {loss_train_ep:.4}, Tr Accuracy = {acc_train_ep*100:.2f}%, Val Accuracy = {acc_val_ep*100:.2f}% (best_val_acc = {best_val_acc*100:.4f}%)\n')
    
    if use_scheduler: scheduler.step() 
end_train = time.time()
print("Train completed in {:.2f} seconds".format(end_train - start_train))

start_test = time.time()
model.load_state_dict(torch.load(model_path, weights_only=False))
test_loss, test_acc = validate(model, test_loader, loss_fn) 
end_test = time.time()
print(f"Test set - Loss: {test_loss:.4f}, Accuracy: {test_acc*100:.2f}%")
print("Test completed in {:.2f} seconds".format(end_test - start_test))

# %% Plot Accuracy and Loss
if plot_results:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(tot_epochs), loss_train, 'b-', label='Train Loss')
    plt.plot(range(tot_epochs), loss_val, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs. Val Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(tot_epochs), acc_train, 'b-', label='Train Accuracy')
    plt.plot(range(tot_epochs), acc_val, 'r-', label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs. Val Accuracy')
    plt.ylim(0.5,1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()