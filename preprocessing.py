import os
import mne
import numpy as np
import matplotlib.pyplot as plt

filepath_abnormal = "./data/abnormal/"
filepath_normal = "./data/normal/"

file_list_abnormal = os.listdir(filepath_abnormal)
file_list_normal = os.listdir(filepath_normal)


def load_signal(file, i):
    eeg_raw = mne.io.read_raw_edf(file)
    channels = eeg_raw.ch_names[0:20]
    eeg_data = eeg_raw.load_data()
    eeg_data.set_eeg_reference('average', projection=True)
    eeg_data.apply_proj()
    eeg_filter_1 = eeg_data.filter(l_freq=0.1, h_freq=None)
    eeg_filter_2 = eeg_filter_1.notch_filter(freqs=60)
    signal = eeg_filter_2.resample(sfreq=100)
    df = signal.to_data_frame(picks=channels)
    matrix = df.to_numpy()
    np.save(f"data{i}", matrix)
    return matrix


def prepare_data(matrix):


    return tensor

























