# %%  08_net_train using my raw data, separatring only the test at the beginning
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from scipy.signal import butter, filtfilt, iirnotch, resample, find_peaks
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split

# File paths
relax_path_1  = "D:\Student_Projects\Thesis Gioele\Raw_data\sub-Gio\ses-S002\\npy\\all_relax.npy"
stroop_path_1 = "D:\Student_Projects\Thesis Gioele\Raw_data\sub-Gio\ses-S003\\npy_stress\\all_stroop.npy"
relax_path_2  = "D:\Student_Projects\Thesis Gioele\Raw_data\sub-Gio\ses-S004\\npy\\all_relax.npy"
stroop_path_2 = "D:\Student_Projects\Thesis Gioele\Raw_data\sub-Gio\ses-S005\\npy_stress\\all_stroop.npy"
dir_r = "D:\Student_Projects\Thesis Gioele\Raw_data\sub-Gio\ses-S007\\all_relax.npy"
dir_s = "D:\Student_Projects\Thesis Gioele\Raw_data\sub-Gio\ses-S007\\all_stress.npy"
dir_r8 = "D:\Student_Projects\Thesis Gioele\Raw_data\sub-Gio\ses-S008\\all_relax.npy"
dir_s8 = "D:\Student_Projects\Thesis Gioele\Raw_data\sub-Gio\ses-S008\\all_stroop.npy"
dir_r9 = "D:\Student_Projects\Thesis Gioele\Raw_data\sub-Gio\ses-S009\\all_relax.npy"
dir_s9 = "D:\Student_Projects\Thesis Gioele\Raw_data\sub-Gio\ses-S009\\all_stroop.npy"

# Load arrays
stroop_data_1 = np.load(stroop_path_1)
relax_data_1  = np.load(relax_path_1)
stroop_data_2 = np.load(stroop_path_2)
relax_data_2  = np.load(relax_path_2)
stroop_data_3 = np.load(dir_s)
relax_data_3  = np.load(dir_r)
stroop_data_4 = np.load(dir_s8)
relax_data_4  = np.load(dir_r8)
stroop_data_5 = np.load(dir_s9)
relax_data_5  = np.load(dir_r9)

# Sampling rate (Hz)
fs_old = 125 # OpenBCI fs
fs_new = 125 # If needed
fs = fs_new

# number of samples for testing
test_samples = 50 * fs_old

# TESTING DATA
stroop_test_raw = stroop_data_5 
relax_test_raw  = relax_data_5 

# 2) Concatenate rest per train+val
stroop_rest = np.concatenate((stroop_data_1, stroop_data_2, stroop_data_3, stroop_data_4), axis=0) 
relax_rest  = np.concatenate((relax_data_1, relax_data_2, relax_data_3, relax_data_4), axis=0) 

# Preprocessing function (ICA, filtering, resample)
def clean_signal_ica(raw_data: np.ndarray,
                     fs: float,  # Hz
                     segment_duration: float = 2.0,  # s
                     peak_rate_max: float = 2,  # Hz = 2 blinks per second
                     kurtosis_thresh: float = 12.0,
                     ica_n_components: int = None) -> np.ndarray:
    """
    Perform ICA-based cleanup of multi-channel signal.

    Input shape must be (timeframes x channels). If the input shape is (channels x timeframes),
    it is automatically transposed.
    """
    transposed = False
    if raw_data.shape[0] < raw_data.shape[1]: 
        raw_data = raw_data.T
        transposed = True

    n_samples, n_channels = raw_data.shape

    if ica_n_components is None:
        ica_n_components = n_channels

    samples_per_segment = int(segment_duration * fs)
    n_segments = int(np.ceil(n_samples / samples_per_segment))
    cleaned = np.zeros_like(raw_data)

    for seg_idx in range(n_segments):
        start = seg_idx * samples_per_segment
        end = min(start + samples_per_segment, n_samples)
        segment = raw_data[start:end, :].T  # shape (channels, segment_length)

        # ICA decomposition
        ica = FastICA(n_components=ica_n_components, random_state=0)
        sources = ica.fit_transform(segment.T).T  # (n_components, n_time)

        # Check how many iterations were used
        if ica.n_iter_ >= 200: # 200 is the default max_iter in FastICA. If it reaches this threshold, it means it did not converge
            print(f"Segment {seg_idx}: ICA did not converge (n_iter_ = {ica.n_iter_}). Skipping artifact removal.")
            cleaned[start:end, :] = raw_data[start:end, :]
            continue

        else: # ICA converged, proceed with the artifact removal
            # collect dropped components and their criteria
            drops = []  # list of (component_idx, criterion)

            for ic in range(sources.shape[0]):
                src = sources[ic]

                # Mid-band peak-rate 
                nyq = 0.5 * fs
                b, a = butter(4, 15 / nyq, btype='low')
                filt = filtfilt(b, a, src) # filtering the component to remove multiple peaks
                threshold = 0.5 * np.max(np.abs(filt))
                peaks, _ = find_peaks(np.abs(filt), height=threshold) 
                rate = len(peaks) / (filt.size / fs)

                # Kurtosis artifact
                k = abs(kurtosis(src))
                if k > kurtosis_thresh and rate < peak_rate_max:
                    drops.append((ic, 'Kurtosis'))

            # Zero-out dropped components and reconstruct
            if drops:
                drop_indices = [ic for ic, _ in drops]
                sources[drop_indices] = 0
            recon = ica.inverse_transform(sources.T).T  # shape (channels, time)
            cleaned[start:end, :] = recon.T  # shape (time, channels)

    if transposed:
        cleaned = cleaned.T

    return cleaned

def preprocess_signal(data, fs_old=125, fs_new=125, 
                      bp_low=1., bp_high=40., bp_order=4,
                      notch_freq=50., notch_q=30.):
    n_samps, n_ch = data.shape
    
    # Z-normalization    
    data_norm = (data - data.mean(axis=0)) / data.std(axis=0)
    # Band & notch filters
    nyq = 0.5 * fs_old
    low, high = bp_low/nyq, bp_high/nyq
    b_bp, a_bp = butter(bp_order, [low, high], btype='band')
    b_notch, a_notch = iirnotch(notch_freq, notch_q, fs_old)
    data_filt = np.zeros_like(data_norm)
    for ch in range(n_ch):
        tmp = filtfilt(b_bp, a_bp, data_norm[:, ch])
        data_filt[:, ch] = filtfilt(b_notch, a_notch, tmp)

    data_ica = clean_signal_ica(data_filt, fs=fs_old)
    
    # Resample
    if fs_old != fs_new:
        n_new = int(round(n_samps * fs_new / fs_old))
        return resample(data_ica, n_new, axis=0)
    return data_ica

# Funzione che estrae sliding windows di 2 s e applica preprocess su ciascuna
def sliding_preproc_subblocks(data_raw: np.ndarray,
                              sub_secs: float,
                              fs: int,
                              overlap_frac: float,
                              label: int) -> TensorDataset:
    sub_size = int(sub_secs * fs)
    step     = int(sub_size * (1.0 - overlap_frac))
    assert step > 0, "overlap_frac deve essere < 1.0"
    subs = []
    for start in range(0, len(data_raw) - sub_size + 1, step):
        seg = data_raw[start:start + sub_size, :]               # [sub_size, n_ch]
        seg_p = preprocess_signal(seg, fs_old=fs, fs_new=fs)    # preprocess su 2 s
        subs.append(seg_p.astype(np.float32))
    if not subs:
        raise ValueError("Nessuna finestra generata: controlla dimensioni e overlap")
    sub_tensor = torch.from_numpy(np.stack(subs))               # [N_sub, sub_size, n_ch]
    labels     = torch.full((len(subs),), label, dtype=torch.long)
    return TensorDataset(sub_tensor, labels)

# Parametri finestre
sub_secs       = 2.0
overlap_train  = 0.5   # 50% overlap per train/val
overlap_test   = 0.75  # 75% overlap per test

# Creo i dataset direttamente da raw + preprocess a 2 s
relax_train_ds  = sliding_preproc_subblocks(relax_rest,    sub_secs, fs, overlap_train, label=0)
stroop_train_ds = sliding_preproc_subblocks(stroop_rest,   sub_secs, fs, overlap_train, label=1)
relax_test_ds   = sliding_preproc_subblocks(relax_test_raw, sub_secs, fs, overlap_test,  label=0)
stroop_test_ds  = sliding_preproc_subblocks(stroop_test_raw,sub_secs, fs, overlap_test,  label=1)

# Split train vs val
total_relax = len(relax_train_ds)
val_len_r   = int(0.1 * total_relax)
train_len_r = total_relax - val_len_r
relax_train, relax_val = random_split(relax_train_ds, [train_len_r, val_len_r],
                                      generator=torch.Generator().manual_seed(42))

total_stroop = len(stroop_train_ds)
val_len_s    = int(0.1 * total_stroop)
train_len_s  = total_stroop - val_len_s
stroop_train, stroop_val = random_split(stroop_train_ds, [train_len_s, val_len_s],
                                        generator=torch.Generator().manual_seed(42))

# Concat e DataLoader
train_ds = ConcatDataset([relax_train, stroop_train])
val_ds   = ConcatDataset([relax_val,   stroop_val])
test_ds  = ConcatDataset([relax_test_ds, stroop_test_ds])

batch_size = 32
train_loader = DataLoader(train_ds,  batch_size=batch_size, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,    batch_size=batch_size, shuffle=True,  drop_last=False)
test_loader  = DataLoader(test_ds,   batch_size=batch_size, shuffle=True,  drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class EEGNet(nn.Module):
    def __init__(self, num_channels, segment_length, num_class=2):
        super().__init__()

        # keep basic hyper-params
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

# Model instantiation
tot_epochs = 40
num_channels = relax_data_1.shape[1]
segment_length = int(sub_secs * fs)
model = EEGNet(num_channels=num_channels, segment_length=segment_length).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
LR = 0.00004

print("LR:", LR)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tot_epochs)

# 4) Training / validation loop
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(1, tot_epochs+1):
    # ——— TRAINING ———
    model.train()
    train_loss = 0.0
    train_acc  = 0.0
    for X, y in train_loader:
        # X: [B, T, C] -> reorder for Conv2d: [B, 1, C, T]
        X = X.permute(0, 2, 1).unsqueeze(1).to(device)  
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        # accumulate
        train_loss += loss.item() * X.size(0)
        train_acc  += (logits.argmax(dim=1) == y).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc  /= len(train_loader.dataset)
    train_losses.append(train_loss) 
    train_accs.append(train_acc)

    # ——— VALIDATION ———
    model.eval()
    val_loss = 0.0
    val_acc  = 0.0
    with torch.no_grad():
        for X, y in val_loader:
            X = X.permute(0, 2, 1).unsqueeze(1).to(device)
            y = y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)

            val_loss += loss.item() * X.size(0)
            val_acc  += (logits.argmax(dim=1) == y).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc  /= len(val_loader.dataset)
    val_losses.append(val_loss) 
    val_accs.append(val_acc)


    # ——— SCHEDULER STEP ———
    if scheduler:
        scheduler.step()

    # ——— LOG ———
    print(f"Epoch {epoch:03d} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

# 5) After training, run test set
model.eval()
test_acc = 0.0
with torch.no_grad():
    for X, y in test_loader:
        X = X.permute(0, 2, 1).unsqueeze(1).to(device)
        y = y.to(device)
        logits = model(X)
        test_acc += (logits.argmax(dim=1) == y).sum().item()
test_acc /= len(test_loader.dataset)
print(f"\nTest Accuracy: {test_acc:.4f}")

# %% plot training and validation loss and acuracy
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Training and Validation Loss and Accuracy', fontsize=16)

# Plot training and validation loss
ax1.plot(train_losses, label='Training Loss')
ax1.plot(val_losses, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()

# Plot training and validation accuracy
ax2.plot(train_accs, label='Training Accuracy')
ax2.plot(val_accs, label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()