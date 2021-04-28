import pickle
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, pkl_file):
        data = pickle.load(pkl_file)
        self.chars = data['chars']
        self.words = data['words']
        self.labels = data['labels']
    
    def __getitem__(self, item):
        return (self.chars[item], self.words[item], self.labels[item])

    def __len__(self):
        assert len(self.chars) == len(self.words)
        return len(self.chars)

