import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
import torchsummary
from PIL import Image

import os
import mne
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset


class EEGSignalDataset(Dataset):
    def __init__(self, train=True):
        self.data = data

    def __getitem__(self, i):
          raw_data = data[:, :, i]
          # raw_data2 = torch.from_numpy(raw_data.astype(np.float32))
          return raw_data

    def __len__(self):
        return data.shape[-1]


# dataset = EEGSignalDataset(trainset)
# dataloader = DataLoader(dataset, batch_size=opt.batch_size, \
#                         num_workers=opt.n_cpu, shuffle=True)
# print(len(dataset))


def load_signal(file):
    eeg_raw = mne.io.read_raw_edf(file)
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
    epochs = size[1]//600
    # freq = 40
    new_signal = np.zeros((epochs, 19, 600))
    for i in range(epochs):
        new_signal[i, :, :] = matrix[:, i*600:(i*600)+600]
    return new_signal

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch

file_list = os.listdir('./data/datasets/')
new_data = []
for file in file_list:
  data = torch.load('./data/datasets/' + file)
  new_data.append(data.numpy())
suma = np.concatenate(new_data, axis=-1)
train_data = EEGSignalDataset(suma)
train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True,
    num_workers=1
)
from numba import njit
from scipy import signal


def cwt_tensor(x, freq, w, widths, Fs ):
    batch_size, N_times, N_chans = x.shape
    cwtm = np.zeros((batch_size, N_chans-1, len(widths), N_times//10))
    for b in range(batch_size):
        for chan in range(N_chans-1):
            cwt_tmp = np.abs(signal.cwt(x[b,chan,:].numpy() , signal.morlet2, widths, w=w))

            im = Image.fromarray(cwt_tmp).resize((N_times//10,len(widths)))
            cwtm[b,chan,:,:] = np.array(im)
    return cwtm
Fs = 100
w=15
freq = np.linspace(1, 40, 40)
widths = w*Fs / (2*freq*np.pi)
import time

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(19, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(896, 128) # mat1 and mat2 shapes cannot be multiplied (16x9472 and 288x128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        #print('In')
        #plt.pcolormesh(t, freq, x[1,1].cpu().numpy(), cmap='viridis', shading='gouraud')
        #plt.show()
        x = x.to(device)
        #print('in:', x.size())
        x = F.relu(self.conv1(x))
        #print('after 1:', x.size())
        x = F.relu((self.conv2(x)))
        #print('after 2:', x.size())
        x = F.relu(self.conv3(x))
        #print('after 3:', x.size()) #after 3: torch.Size([16, 32, 4, 74])
        x = torch.flatten(x, start_dim=1)
        #print('after flat:', x.size())
        x = F.relu(self.linear1(x)) # wyjście z tej warstwy jest rozdwojone i jedna kopia idzie do liczenia mu a druga do sigmy
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x)) # JZ tu jeszcze nie do konca rozumiem czemu sigma jest liczona przez exp
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum() # liczymy Kulback-Leiibler term do f. koksztu
        return z

class Decoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128,896 ),#3* 3 * 32
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 4, 7)) # 32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=(1,0)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 19, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        #print('unf:', x.size())
        x = self.decoder_conv(x)
        #print('dec:', x.size())
        #x = torch.sigmoid(x)
        #plt.pcolormesh( x[1,1].detach().cpu().numpy(), cmap='viridis', shading='gouraud') # t, freq,
        #plt.show()
        #print('Out')
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)

### Set the random seed for reproducible results
torch.manual_seed(0)

d = 20

vae = VariationalAutoencoder(latent_dims=d)

lr = 1e-3

optim = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

vae.to(device)

### Training function
from tqdm.notebook import trange, tqdm
def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    items = 0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for _, x in enumerate(tqdm(dataloader)):
        print ("SHAPE:", x.shape)
        # Move tensor to the proper device
        cwtm = torch.Tensor( cwt_tensor(x, freq, w, widths, Fs ))
        x = F.normalize(cwtm, dim = 3)


        x = x.to(device)
        x_hat = vae(x)



        #print(x_hat)
        # Evaluate loss
        loss = ((x - x_hat)**2).sum() #+ vae.encoder.kl
        #print('1 ',loss.item())
        #break

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()
        items +=1
    print('In')
    plt.pcolormesh( x[1,1].detach().cpu().numpy(), cmap='viridis', shading='gouraud') # t, freq,
    plt.show()
    print('Out', x_hat.size())
    plt.pcolormesh( x_hat[1,1].detach().cpu().numpy(), cmap='viridis', shading='gouraud') # t, freq,
    plt.show()



    return train_loss / items


def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    items = 0
    with torch.no_grad(): # No need to track the gradients
        for _, x in dataloader:
            # Move tensor to the proper device
            cwtm = torch.Tensor( cwt_tensor(x, freq, w, widths, Fs ))

            x = F.normalize(cwtm, dim = 3)
            x = x.to(device)
            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() #+ vae.encoder.kl # suma bl. średniokwadratowego i mieary KL
            val_loss += loss.item()
            items+=1
    print('In test')
    plt.pcolormesh( x[1,1].detach().cpu().numpy(), cmap='viridis', shading='gouraud') # t, freq,
    plt.show()
    print('Out test', x_hat.size())
    plt.pcolormesh( x_hat[1,1].detach().cpu().numpy(), cmap='viridis', shading='gouraud') # t, freq,
    plt.show()



    return val_loss / items


num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train_epoch(vae,device,train_loader,optim)
    print('\n EPOCH {}/{} \t train loss {:.3f}'.format(epoch + 1, num_epochs,train_loss))