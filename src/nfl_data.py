import torch
from torch.utils.data import Dataset

class NFLDataset(Dataset):
    def __init__(self, data_path: str=None):
        
        self.data = torch.load(data_path) # data is a dictionary. Keys are weeks,
                                   # values are dictionaries whose keys are
                                   # gpids and values are tuples of (features, labels) tensor objects each.
        self.data = [play_tup for week_dict in self.data.values() for play_tup in week_dict.values()]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index] # might need to modify this