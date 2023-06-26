import os
import mne
import numpy as np
import matplotlib.pyplot as plt


def load_signal(file):
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
    return matrix


def prepare_dataset(matrix):
    size = matrix.shape()
    epochs = size[1]//600
    freq = 40
    new_signal = np.zeros((epochs, 19, freq, 600))
    for i in range(epochs):
        new_signal[i, :, :, :] = matrix[:, i*600, (i*600)+600]

    return new_signal


























