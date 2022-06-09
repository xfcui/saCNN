import h5py
from torch.utils.data.dataset import Dataset


class MyDataset(Dataset):
    def __init__(self, filename):
        with h5py.File(filename, 'r') as f:
            self.data = f['data'][()]

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def __len__(self):
        return len(self.data)
