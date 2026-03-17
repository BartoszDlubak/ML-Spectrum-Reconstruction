import numpy as np
import torch
from torch.utils.data import Dataset

from .process import process_real, process_simulated
from .normaliser import minmaxNormaliser



class spectraDataset(Dataset):

    def __init__(self, wavelengths, observed_data, cond_mask, observed_mask, gt_mask):
        self.timepoints = torch.tensor(wavelengths, dtype=torch.float32)
        self.observed_data = torch.tensor(observed_data, dtype=torch.float32).unsqueeze(1)
        self.cond_mask = torch.tensor(cond_mask, dtype=torch.float32).unsqueeze(1)
        self.observed_mask = torch.tensor(observed_mask, dtype=torch.float32).unsqueeze(1)
        self.gt_mask = torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.observed_data)

    def __getitem__(self, idx):

        return {
            "observed_data": self.observed_data[idx],
            "observed_mask": self.observed_mask[idx],
            "cond_mask": self.cond_mask[idx],
            "gt_mask": self.gt_mask[idx],
            "timepoints": self.timepoints,
        }
 
        
        
def build_dataset(
    wavelengths,
    observed_data, 
    cond_mask, 
    observed_mask, 
    gt_mask,
):
    return spectraDataset(
        wavelengths,
        observed_data, 
        cond_mask, 
        observed_mask, 
        gt_mask
        )