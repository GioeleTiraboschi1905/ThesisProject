# Applies z-normalization, BPF, notch filtering to raw EEG data
# %%
import os
import glob
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split

# %%
# set to false to not plot the signals
plot_data = True

# put False to avoid the Linear Surrogating Technique for data augmentation #
use_surrogate = False

# Data loading
path = "D:\\Student_Projects\\Thesis Gioele\\00_04_RAW EEG STRESS DATASET"
relax_files = glob.glob(os.path.join(path, "Relax", "*.csv"))
stroop_files = glob.glob(os.path.join(path, "Stroop", "*.csv"))
arithmetic_files = glob.glob(os.path.join(path, "Arithmetic", "*.csv"))
mirror_files = glob.glob(os.path.join(path, "Mirror_image", "*.csv"))

# loading CSV files into a list of DataFrames
def load_csv_files(file_list):
    data_list = []
    for file in file_list:
        df = pd.read_csv(file) 
        df = df.drop(df.columns[0], axis=1) # Remove the first column (index of timeframes)
        data_list.append(df.values)
    return data_list

relax_data = load_csv_files(relax_files)
stroop_data = load_csv_files(stroop_files)
arithmetic_data = load_csv_files(arithmetic_files)
mirror_data = load_csv_files(mirror_files)

fs = 128

start_filt = time.time()

# z-normalisation
def z_normalization(data_list):
    for i in range(len(data_list)): 
        data_list[i] = (data_list[i] - data_list[i].mean()) / np.std(data_list[i]) # subtract avg, divide by std
    return data_list

relax_data_avg = z_normalization(relax_data.copy())
stroop_data_avg = z_normalization(stroop_data.copy())
arithmetic_data_avg = z_normalization(arithmetic_data.copy())
mirror_data_avg = z_normalization(mirror_data.copy())

# %%  Filtering
def bandpass_filter(data, lowcut=1, highcut=40, fs=128, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def notch_filter(data, fs=128, freq=50, quality_factor=30): 
    b, a = iirnotch(freq, quality_factor, fs)
    return filtfilt(b, a, data, axis=0)

def preprocess_data(data_list): # passband + notch filter
    return [notch_filter(bandpass_filter(df, lowcut=1, highcut=40, fs=fs), fs=fs) for df in data_list]

relax_data_filt = np.stack(preprocess_data(relax_data_avg), axis=0)
stroop_data_filt = np.stack(preprocess_data(stroop_data_avg), axis=0)
arithmetic_data_filt = np.stack(preprocess_data(arithmetic_data_avg), axis=0)
mirror_data_filt = np.stack(preprocess_data(mirror_data_avg), axis=0)

end_filt = time.time()
print("Data filtered in ", end_filt-start_filt, "seconds")

# %% Data Augmentation using Linear Surrogating Technique
if use_surrogate:
    def linear_surrogate(signal_tensor):
        fft_signal = torch.fft.fft(signal_tensor)
        magnitude = torch.abs(fft_signal)         # Module
        phase = torch.angle(fft_signal)             # Original phase
        random_phase = torch.rand_like(phase) * 2 * torch.pi  # New random phase
        fft_surrogate = magnitude * torch.exp(1j * random_phase)  # New FFT with the random phase
        surrogate_signal = torch.fft.ifft(fft_surrogate).real     # IFFT (take the real part)
        return surrogate_signal

    def augment_linear_surrogating(data_list):
        augmented_data = []
        for subject_array in data_list: 
            surrogate_array = np.zeros_like(subject_array)

            for ch in range(subject_array.shape[1]):
                signal_tensor = torch.tensor(subject_array[:, ch].copy(), dtype=torch.float32)
                surrogate_signal = linear_surrogate(signal_tensor)
                surrogate_array[:, ch] = surrogate_signal.numpy()
            augmented_data.append(surrogate_array)  
        return augmented_data

    # Splitting into training and testing sets, then applying data augmentation
    def split_and_augment(data):
        train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)
        train_augmented = augment_linear_surrogating(train_data)
        train_doubled = np.concatenate((np.array(train_data), np.array(train_augmented)), axis=0)
        return train_doubled, test_data

    relax_train_doubled, relax_test = split_and_augment(relax_data_filt)
    stroop_train_doubled, stroop_test = split_and_augment(stroop_data_filt)
    arithmetic_train_doubled, arithmetic_test = split_and_augment(arithmetic_data_filt)
    mirror_train_doubled, mirror_test = split_and_augment(mirror_data_filt)

    relax_data_filt = relax_train_doubled
    stroop_data_filt = stroop_train_doubled
    arithmetic_data_filt = arithmetic_train_doubled
    mirror_data_filt = mirror_train_doubled

    relax_all = np.concatenate((relax_data_filt, relax_test), axis=0)
    stroop_all = np.concatenate((stroop_data_filt, stroop_test), axis=0)
    arithmetic_all = np.concatenate((arithmetic_data_filt, arithmetic_test), axis=0)
    mirror_all = np.concatenate((mirror_data_filt, mirror_test), axis=0)

else: # no data augmentation   (full dataset)
    relax_all = relax_data_filt.copy()
    stroop_all = stroop_data_filt.copy()
    arithmetic_all = arithmetic_data_filt.copy()
    mirror_all = mirror_data_filt.copy()

    # train - test splitting
    relax_data_filt, relax_test = train_test_split(relax_data_filt, test_size=0.15, random_state=42)
    stroop_data_filt, stroop_test = train_test_split(stroop_data_filt, test_size=0.15, random_state=42)
    arithmetic_data_filt, arithmetic_test = train_test_split(arithmetic_data_filt, test_size=0.15, random_state=42)
    mirror_data_filt, mirror_test = train_test_split(mirror_data_filt, test_size=0.15, random_state=42)

# %% Data Saving and plotting ...
start_save = time.time()

folder_path = "D:\\Student_Projects\\Thesis Gioele\\Codes\\Data\\00\\00_"
           
end_path_tr = "_ica_filtered_train.npy"
end_path_test = "_ica_filtered_test.npy"

# Save the filtered data
# train
np.save(folder_path + "relax" + end_path_tr, relax_data_filt)
np.save(folder_path + "stroop" + end_path_tr, stroop_data_filt)
np.save(folder_path + "arithmetic" + end_path_tr, arithmetic_data_filt)
np.save(folder_path + "mirror" + end_path_tr, mirror_data_filt)
# test
np.save(folder_path + "relax" + end_path_test, relax_test)
np.save(folder_path + "stroop" + end_path_test, stroop_test)
np.save(folder_path + "arithmetic" + end_path_test, arithmetic_test)
np.save(folder_path + "mirror" + end_path_test, mirror_test)

# save the full dataset 
end_path = "_full_filtered.npy"
np.save(folder_path + "relax" + end_path, relax_all)
np.save(folder_path + "stroop" + end_path, stroop_all)
np.save(folder_path + "arithmetic" + end_path, arithmetic_all)
np.save(folder_path + "mirror" + end_path, mirror_all)

end_save = time.time()
print("Filtered data saved in", end_save-start_save, "seconds")
print("Total time (s):", end_filt-start_filt + end_save-start_save)

# %% Plot signal befor and after filtering (in samples)
if plot_data:
    subject_index = 10
    channel_indices = [3, 5, 17, 9]
    shift_amount = 200  # vertical shift to separate channels
    channel_names = ['F7', 'FC1', 'Oz', 'T7'] # Nomi dei canali corrispondenti agli indici
    time_axis = np.arange(len(relax_data[subject_index][:, 0])) / fs

    plt.figure(figsize=(15, 10))

    # Plot original data
    plt.subplot(2, 1, 1)
    for i, channel_index in enumerate(channel_indices):
        plt.plot(time_axis, relax_data[subject_index][:, channel_index] + i * shift_amount, label=channel_names[i])
    plt.title('Raw EEG Data (Relax)')
    plt.xlabel('Time (s)')
    plt.ylabel('Shifted Amplitude (uV)')
    plt.xlim(0, 75)
    plt.legend(loc='lower right')

    # Plot after filtering
    shift_amount = 1.5
    plt.subplot(2, 1, 2)
    for i, channel_index in enumerate(channel_indices):
        plt.plot(time_axis, relax_data_filt[subject_index][:, channel_index] + i * shift_amount, label=channel_names[i])
    plt.title('EEG Data After Filtering (Relax)')
    plt.xlabel('Time (s)')
    plt.ylabel('Shifted Amplitude (uV)')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()