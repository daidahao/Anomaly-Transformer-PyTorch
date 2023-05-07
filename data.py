import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Dataset:
    def __init__(self, path: str, n_features: int, window_size: int,
        step_size: int=1, stride: int=1, skiprows: int=0):
        # data parameters
        self.step_size = step_size
        self.window_size = window_size
        self.stride = stride
        # read csv
        df = pd.read_csv(path)
        # read timestamps and labels
        self.ts = df.values[skiprows::stride, 0]
        self.labels = df.values[skiprows::stride, -1]
        # read data
        self.normaliser = StandardScaler()
        data = df.values[skiprows::stride, 1:n_features+1]
        self.data = self.normaliser.fit_transform(data)

        print(f'Loaded {len(self.data)} samples from {path}')
        print(f'Number of features: {len(self.data[0])}')
    
    def __len__(self):
        return (len(self.data) - self.window_size) // self.step_size + 1
    
    def __getitem__(self, index):
        i_start = index * self.step_size
        i_end = i_start + self.window_size
        return self.data[i_start:i_end], self.labels[i_end-1]




