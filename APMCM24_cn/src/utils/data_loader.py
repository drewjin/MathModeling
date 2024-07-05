import torch 
import torch.nn as nn
from torch.utils.data import Dataset

__all__ = ['Dataset_2']

class Dataset_2(Dataset):
    def __init__(self, data, label, cls_label):
        self.data = data.to(torch.float32)
        self.label = label.to(torch.float32)
        self.cls_label = cls_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.label[idx]
        cls_label = self.cls_label[idx]
        
        return features, (label, cls_label)