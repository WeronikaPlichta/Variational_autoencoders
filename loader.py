import torch
from torch.utils.data import DataLoader, Dataset


class EEGSignalDataset(Dataset):
    def __init__(self, data, gt, m=m, s=s, soft_label=True, train=True):
        self.data = data
        self.gt = gt
        self.train = train
        self.soft_label = soft_label
        self.eps = 1e-7
        if train:
            self.index = resample_data(gt)
        else:
            self.index = [(i, j) for i in range(len(data)) for j in range(data[i].shape[1])]
        for dt in self.data:
            dt -= m
            dt /= s + self.eps

    def __getitem__(self, i):
        i, j = self.index[i]
        raw_data, label = self.data[i][:, max(0, j - opt.in_len + 1):j + 1], \
            self.gt[i][:, j]

        pad = opt.in_len - raw_data.shape[1]
        if pad:
            raw_data = np.pad(raw_data, ((0, 0), (pad, 0)), 'constant', constant_values=0)

        raw_data, label = torch.from_numpy(raw_data.astype(np.float32)), \
            torch.from_numpy(label.astype(np.float32))
        if self.soft_label:
            label[label < .02] = .02
        return raw_data, label

    def __len__(self):
        return len(self.index)


dataset = EEGSignalDataset(trainset, gt)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, \
                        num_workers=opt.n_cpu, shuffle=True)
print(len(dataset))