import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset



class DisturbanceDataset(Dataset):
    def __init__(self, x, y):


        # Load data 
        #with open('./data_estimator.npy', 'rb') as f:
        #    self.x = np.load(f)
        #    self.y = np.load(f)
        
        self.x = x
        self.y = y
        
        size_x = self.x.shape[0]
        size_y = self.y.shape[0]
        self.y = self.y.reshape((size_y, 1, 6))

        # Normalize data
        self.x[:, :, 27:33] = self.x[:, :, 27:33]/200.
        self.x = self.x[:].reshape(self.x.shape[0], self.x.shape[1]*self.x.shape[2])
        self.y = self.y/20.
        


    def __len__(self):
        return self.x.shape[0]


    def __getitem__(self, idxs):
        if torch.is_tensor(idxs):
            idxs = idxs.tolist()

        x = self.x[idxs]
        y = self.y[idxs]


        #return sample
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()







