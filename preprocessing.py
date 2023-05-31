import os
import mne
import numpy as np
import matplotlib.pyplot as plt

filepath_abnormal = "./data/abnormal/"
filepath_normal = "./data/normal/"

file_list_abnormal = os.listdir(filepath_abnormal)
file_list_normal = os.listdir(filepath_normal)


def load_signal(file):
    eeg_raw = mne.io.read_raw_edf(file)
    mont = mne.channels.make_standard_montage('standard_1020')
    eeg_mont = eeg_raw.copy().set_montage(mont)

    return signal


def prepare_data(signal):
    return tensor
