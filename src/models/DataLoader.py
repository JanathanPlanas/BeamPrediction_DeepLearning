
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'




class Loaders():

    def __init__(self, 
                 dimension :int ,
                 file_path='future_clean/future-1/',
                 batch_size = None) -> None:


        self.file_path = file_path
        data_paths = ['train.parquet', 'val.parquet', 'test.parquet']
        data_dict = {}

        for data_type, data_path in zip(['train', 'val', 'test'], data_paths):
            df = pd.read_parquet(f'{self.file_path }{data_path}')
            data_dict[data_type] = df.to_numpy()

        self.train = (data_dict['train'])
        self.val = (data_dict['val'])
        self.test = (data_dict['test'])

        if normalization == True:
            X_train = normalize(self.train[:, :-dimension])
            y_train = self.train[:, -dimension:]
            X_test = normalize(self.test[:, :-dimension])
            y_test = self.test[:, -dimension:]

        if normalization == False:
            X_train = (self.train[:, :-dimension])
            y_train = self.train[:, -dimension]
            X_test = (self.test[:, :-dimension])
            y_test = self.test[:, -dimension]


        numpy_list = [X_train, y_train, X_test, y_test]
        tensor_list = []
        self.numpy_list = numpy_list

        for npy in numpy_list:
            tensor = torch.Tensor(npy).to(device, dtype=torch.double)
            tensor_list.append(tensor)

        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = tensor_list

        datasets = [train_dataset, test_dataset] = [
            TensorDataset(X_train_tensor, y_train_tensor),
            TensorDataset(X_test_tensor, y_test_tensor)
        ]

        loaders = [train_loader, test_loader] = [
            DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            for dataset, shuffle in zip(datasets, [True, True])
        ]

        self.train_loader , self.test_loader = loaders