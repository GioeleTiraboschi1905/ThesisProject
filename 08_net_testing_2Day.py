# VALIDATE USING A PRE-TRAINED MODEL, testing on data filtered in the 2-seconds segments
# %%
import time
import math
import torch
import numpy as np
import torch.nn as nn
from scipy.stats import kurtosis
from sklearn.utils import shuffle
from sklearn.decomposition import FastICA
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from scipy.signal import butter, filtfilt, iirnotch, resample, find_peaks

# %%
start = time.time()
fs_old = 125 # recorded sampling frequency
fs = 125     # sampling frequency of the data used for the training of the model
num_channels = 16
factor = 2   # seconds of data for each segment
segment_length = factor * fs 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
dir_r = "D:\Student_Projects\Thesis Gioele\Raw_data\sub-Gio\ses-S007\\all_relax.npy"
dir_s = "D:\Student_Projects\Thesis Gioele\Raw_data\sub-Gio\ses-S007\\all_stress.npy"
relax_data = np.load(dir_r)
stress_data = np.load(dir_s)
relax_data = relax_data[-2*fs_old:] # in this case, I use only the last 2 seconds of data
stress_data = stress_data[-2*fs_old:]

end_loading =time.time()
print(f"Data loaded in {end_loading - start:.4f} seconds")

# %% Resampling
if fs != fs_old:
    def resample_data(data, orig_fs=125, target_fs=128):
        """
        Resample along time-axis (0), preserving shape:
        data: (n_samples, n_channels)
        """
        n_samples, n_ch = data.shape
        new_n = int(n_samples * (target_fs / orig_fs))
        data_rs = resample(data, new_n, axis=0)
        return data_rs

    relax_rs  = resample_data(relax_data, orig_fs=fs_old, target_fs=fs)
    stress_rs = resample_data(stress_data, orig_fs=fs_old, target_fs=fs)
else:
    relax_rs = relax_data
    stress_rs = stress_data

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
    
model = EEGNet(num_channels=num_channels, segment_length=segment_length).to(device)

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
                      kurtosis_percentile=80, bp_low=1., bp_high=40., bp_order=4,
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

    # ICA
    data_ica = clean_signal_ica(data_filt, fs=fs_old)

    return data_ica

# Preprocessing on each of the 2-second-long subsegments
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
    
    sub_tensor = torch.from_numpy(np.stack(subs))               # [N_sub, sub_size, n_ch]
    labels     = torch.full((len(subs),), label, dtype=torch.long)
    return TensorDataset(sub_tensor, labels)

sub_secs       = 2.0
overlap_test   = 0.25 

relax_test_ds   = sliding_preproc_subblocks(relax_rs, sub_secs, fs, overlap_test,  label=0)
stroop_test_ds  = sliding_preproc_subblocks(stress_rs,sub_secs, fs, overlap_test,  label=1)

# %%
test_ds  = ConcatDataset([relax_test_ds, stroop_test_ds])

test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=True)
X_batch, y_batch = next(iter(test_loader))

X = X_batch.permute(0, 2, 1).to(device)   # -> [N, 16, 250]
y = y_batch.to(device)

X, y = shuffle(X, y, random_state=42)

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)  # shape: (710, 1, 16, 250)
y_tensor = torch.tensor(y, dtype=torch.long).to(device)   

end_data_preparation = time.time()
print(f"Data preprocessed and prepared for the net in {end_data_preparation - end_loading:.4f} seconds")

# %% VALIDATE USING A PRE-TRAINED MODEL
model_path = "Models\model_trained_data1.pt" 
# otherwise, other trained models are in:  "eegnet_trained.pt" #"D:\\Student_Projects\\Thesis Gioele\\Codes//Data//04//04_model_2cl_EEGNet_16.pt"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.eval()
with torch.no_grad():
    outputs = model(X_tensor)  # shape: (N, num_classes)
    predictions = torch.argmax(outputs, dim=1)  # shape: (N,)
accuracy = (predictions == y_tensor).float().mean().item()

end = time.time()
print(f"Accuracy: {accuracy:.4f}")
print(f"Classification time: {end - end_data_preparation:.4f} seconds")
print(f"Total time: {end-start:.4f} seconds")


# %%
# Parameters for ITR
processing_time = end - start    # seconds for preprocessing + classification
T = 2 + processing_time # 2 seconds segment + processing time 

P = accuracy
N = 2 # num_classes

# Avoid log(0) by adding a small epsilon
epsilon = 1e-10
if P == 1.0:
    itr = math.log2(N) * 60 / T
else:
    itr = (
        math.log2(N) + P * math.log2(P + epsilon) + (1 - P) * math.log2((1 - P + epsilon) / (N - 1))
    ) * 60 / T

# Print results
print(f"ITR: {itr:.2f} bits/min")