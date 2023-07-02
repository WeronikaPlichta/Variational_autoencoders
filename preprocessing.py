import os
import scipy as sig
import mne
import numpy as np
import matplotlib.pyplot as plt

from time_freq import cwt


def load_signal(file):
    eeg_raw = mne.io.read_raw_edf(f"./data/normal/{file}")
    channels = eeg_raw.ch_names[0:19]
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
    size = matrix.shape
    print("MATRIX SHAPE:", matrix.shape)
    epochs = size[0]//600
    new_signal = np.zeros((600, 40, 20, epochs))
    # new_signal = np.zeros((600, 20, epochs))
    freq = np.linspace(1, 40, 100)
    w = 7
    widths = w * 100 / (2 * freq * np.pi)
    for i in range(epochs):
        for j in range(20):
            P = sig.cwt(matrix[i * 600:(i * 600) + 600, j], sig.morlet2, widths, w=w)
            new_signal[:, :, j, i] = P
    return new_signal
