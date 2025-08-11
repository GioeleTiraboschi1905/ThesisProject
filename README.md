# EEG-Based Stress Classification with EEGNet

This repository contains the code for my master's thesis on EEG-based stress classification using EEGNet, a compact CNN designed for EEG signal analysis.  
The model takes preprocessed EEG data and classifies it into stress vs. non-stress states, aiming to support real-time stress monitoring with portable EEG devices.

# Repository Structure

00_pre_processing.py           # Applies z-normalisation, BPF, notch filtering to raw EEG data

01_ICA_v0.py                   # Apply ICA to  filtered EEG Data

04_EEGNet_v2_1.py              # EEGNet implementation to train and classify data 

04_EEGNet_v2_1_kFold.py        # k-fold on: batch size, learning rate, weight decay, betas, use_scheduler

08_net_testing_2Day.py         # VALIDATE USING A PRE-TRAINED MODEL, testing on data filtered in the 2-second segments

08_net_train_new_data_v1.py    # 08_net_train using my raw data, separating only the test at the beginning

# Data
The public dataset was available at the link:
https://figshare.com/articles/dataset/SAM_40_Dataset_of_40_Subject_EEG_Recordings_to_Monitor_the_Induced-Stress_while_performing_Stroop_Color-Word_Test_Arithmetic_Task_and_Mirror_Image_Recognition_Task/14562090?file=27956376
