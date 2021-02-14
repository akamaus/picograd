from torch.utils.data import Dataset
class SubDataset(Dataset):
    """ A wrapper returning subset of indices """
    def __init__(self, ds, indices):
        self._indices = list(indices)
        self._ds = ds

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._ds[self._indices[idx]]
