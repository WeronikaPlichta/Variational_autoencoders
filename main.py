import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from preprocessing import load_signal, prepare_data
from time_freq import cwt


filepath_abnormal = "./data/abnormal/"
filepath_normal = "./data/normal/"

file_list_abnormal = os.listdir(filepath_abnormal)
file_list_normal = os.listdir(filepath_normal)
# number_of_cases, channels, zakres czestości mapki, czas
lista_nowych_plików = []
for file in file_list_normal:
    new_signal = load_signal(file)
    raw_data = prepare_data(new_signal)
    lista_nowych_plików.append(raw_data)
suma = np.concatenate(lista_nowych_plików, axis=0)

train_data = torch.from_numpy(suma.astype(np.float32))
dataloader = DataLoader(train_data, batch_size=8, shuffle=True)


