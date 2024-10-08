import scipy.io
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class UniMiB_dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def combine(self, X, Y, adjust_dims=False):
        self.X = np.concatenate((self.X, X), axis=0)
        self.Y = np.concatenate((self.Y, Y), axis=0)

        if adjust_dims:
            adjust_dims()
    
    def adjust_dims(self):
        if len(self.X.shape) > 3:
            self.X = np.squeeze(self.X, axis=1)
        else:
            self.X = np.expand_dims(self.X, axis=1)
    
    def set_dims(self, to_expand=True):
        if to_expand and len(self.X.shape) == 3:
            self.X = np.expand_dims(self.X, axis=1)
        elif not to_expand and len(self.X.shape) == 4:
            self.X = np.squeeze(self.X, axis=1)

    def __len__(self):
        self.len = len(self.X)
        return self.len
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx,:]).float()
        Y = torch.tensor(self.Y[idx,:]).long()
        
        return X, Y
    
class UniMib_DataManager():
    def __init__(self, dataset_dir, rand_seed=42):
        self.dataset_dir = dataset_dir

        self.data   = None
        self.labels = None
        self.rand_seed = rand_seed
        self.__load_dataset(self.dataset_dir, 'fall', 3, 151)

    def get_datasets(self, 
                     time_step, 
                     input_size, 
                     n_users,
                     client_indices=None,
                     task_type='fall'):
        if self.data is None:
            self.__load_dataset(self.dataset_dir, task_type, time_step, input_size)

        n_classes = len(np.unique(self.labels))

        return self.__get_datasets(n_users, task_type, client_indices=client_indices), n_classes
    

    def __get_datasets(self, num_users, task_type, client_indices=None):
        if self.data is None:
            self.__load_dataset(self.dataset_dir, task_type, 3, 151, True)

        n_data = len(self.labels)
        spliter_fn = StratifiedShuffleSplit(n_splits=num_users, train_size=n_data // num_users, random_state=84)

        train_datasets = {}
        test_datasets  = {}

        for i, (train_idx, test_idx) in enumerate(spliter_fn.split(self.data, self.labels)):
            if client_indices is not None:
                train_idx = client_indices[i]

            train_data, train_labels = self.data[train_idx], self.labels[train_idx]
            stratify = train_labels if client_indices is None else None

            X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=self.rand_seed, stratify=stratify)
            train_datasets[i] = UniMiB_dataset(X_train, Y_train)
            test_datasets[i]  = UniMiB_dataset(X_test, Y_test)
            print(len(X_train), len(X_test))

        return train_datasets, test_datasets

    def __load_dataset(self, dataset_dir, task_type, time_step, input_size):
        data_file_dir  = dataset_dir + f'{task_type}_data.mat'
        label_file_dir = dataset_dir + f'{task_type}_labels.mat'

        data   = scipy.io.loadmat(data_file_dir)[f'{task_type}_data']
        labels = scipy.io.loadmat(label_file_dir)[f'{task_type}_labels']
        
        data   = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels[:, 0], dtype=np.int64)

        self.data = np.reshape(data, (-1, 1, time_step, input_size))
        self.labels = np.reshape(labels, (-1, 1)) - 1
