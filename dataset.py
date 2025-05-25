import torch
from torch.utils.data import Dataset
from config import tokenized_data_path, max_length

class PoemDataset(Dataset):
    def __init__(self):
        self.data = torch.load(tokenized_data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = x[:max_length]
        return {
            'input_ids': x,
            'labels': x.clone()
        }
