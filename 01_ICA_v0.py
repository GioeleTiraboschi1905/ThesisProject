# Apply ICA to  filtered EEG Data
# %%
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from scipy.signal import butter, filtfilt, find_peaks

# %%
# set to True if you want the plot
plot_signal = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA LOADING
folder_path = "D:\\Student_Projects\\Thesis Gioele\\Codes\\Data\\00\\00_"
end_path = "_zNorm_full_filtered_aug.npy"

# Load the filtered data (full dataset)
relax_data = np.load(folder_path + "relax" + end_path, allow_pickle=True)
stroop_data = np.load(folder_path + "stroop" + end_path, allow_pickle=True)
arithmetic_data = np.load(folder_path + "arithmetic" + end_path, allow_pickle=True)
mirror_data = np.load(folder_path + "mirror" + end_path, allow_pickle=True)

fs = 128

# %%
start_ica = time.time() 

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

relax_data_ica = [clean_signal_ica(subject_data, fs=fs) for subject_data in relax_data]
stroop_data_ica = [clean_signal_ica(subject_data, fs=fs) for subject_data in stroop_data]
arithmetic_data_ica = [clean_signal_ica(subject_data, fs=fs) for subject_data in arithmetic_data]
mirror_data_ica = [clean_signal_ica(subject_data, fs=fs) for subject_data in mirror_data]

end_ica = time.time()
print("ICA total time (full dataset): ", end_ica - start_ica) 

######### SALVATAGGIO DEI DATI ICA #########
folder_path = "D:\\Student_Projects\\Thesis Gioele\\Codes\\Data\\01\\01_"
end_path = "_ica_v0.npy"

# Save the filtered data
np.save(folder_path + "relax" + end_path, relax_data_ica)
np.save(folder_path + "stroop" + end_path, stroop_data_ica)
np.save(folder_path + "arithmetic" + end_path, arithmetic_data_ica)
np.save(folder_path + "mirror" + end_path, mirror_data_ica)

# %%
if plot_signal:
    subject_index = 0
    channel_indices = range(3,7)
    shift_amount = 0.9 
    ym = -1.5
    yM = 3.5

    plt.figure(figsize=(15, 10))

    # Plot original data
    plt.subplot(2, 1, 1)
    for i, channel_index in enumerate(channel_indices):
        plt.plot(relax_data[subject_index][:, channel_index] + i * shift_amount, label=f'Ch{channel_index + 1}')
    plt.title('EEG Data Before ICA')
    plt.xlabel('Samples')
    plt.ylabel('Shifted Amplitude')
    plt.ylim(ym, yM)  

    # Plot data after ICA
    plt.subplot(2, 1, 2)
    for i, channel_index in enumerate(channel_indices):
        plt.plot(relax_data_ica[subject_index][:, channel_index] + i * shift_amount, label=f'Ch{channel_index + 1}')
    plt.title('EEG Data After ICA')
    plt.xlabel('Samples')
    plt.ylabel('Shifted Amplitude')
    plt.ylim(ym, yM)  

    plt.tight_layout()
    plt.show()