import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from preprocessing import load_signal, prepare_dataset
from time_freq import cwt


filepath_abnormal = "./data/abnormal/"
filepath_normal = "./data/normal/"

file_list_abnormal = os.listdir(filepath_abnormal)
file_list_normal = os.listdir(filepath_normal)

# number_of_cases, channels, zakres czestości mapki, czas
all_data = []
for file in file_list_normal:
    new_signal = load_signal(file)
    raw_data = prepare_dataset(new_signal)
    all_data.append(raw_data)

suma = np.concatenate(all_data, axis=-1)
print("SUMA SHAPE", suma.shape)
train_data = torch.from_numpy(suma.astype(np.float32))
print(train_data.shape)
dataloader = DataLoader(train_data, batch_size=16, num_workers=1, shuffle=True)

#umap - paczka pythonowa
#obrazki zmniejszać rozdzielczość
#tensory jednej osoby na jeden katalog, czytać katalog po katalogu
#tablica ze ścieżkami w pamięci, ładować jako batch kilka losowych sygnałów wyczytanych z tablicy
#ImageFolder i #DatasetFolder
#klasa Dataset na podstawie tego pytrochowego
#wczytywanie tensorów już wcześniej stworzonych ze ścieżki pliku
#jak skracać to w wymiarze czasu (np. z 600 do 60)
#PIL do zmniejszania mapki, potraktować jako obrazek



