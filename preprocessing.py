import os
import scipy.signal as ss
import mne
import torch
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


def prepare_dataset(matrix, file): #NIE OBRAZKI TYLKO DALEJ SYGNAŁY, NA OBRAZKI BATCHAMI, TAK JAK TO JAREK ZROBIŁ W SWOIM NOTEBOOKU
    size = matrix.shape
    print("MATRIX SHAPE:", matrix.shape)
    epochs = size[0]//600
    # new_signal = np.zeros((40, 60, 20, epochs))
    new_signal = np.zeros((600, 20, epochs))
    # freq = np.linspace(1, 40, 40)
    # w = 7
    # widths = w * 100 / (2 * freq * np.pi)
    for i in range(epochs):
        # for j in range(20):
            # P = ss.cwt(matrix[i * 600:(i * 600) + 600, j], ss.morlet2, widths, w=w)
            # new_signal[:, :, j, i] = np.abs(P[:, ::10])
        new_signal[:, :, i] = matrix[i * 600:(i * 600) + 600, :]
    data = torch.from_numpy(new_signal.astype(np.float32))
    torch.save(data, f"./data/datasets/data_n{file}.pt")
    return new_signal
