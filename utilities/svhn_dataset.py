import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


class SVHN_DataManager():
    def __init__(self, dataset_dir = 'datasets/svhn/') -> None:
        self.dataset_dir = dataset_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.n_classes = 10
        self.__load_dataset()

    def get_datasets(self, num_users, client_indices=None):
        n_data = len(self.labels)
        spliter_fn = StratifiedShuffleSplit(n_splits=num_users, train_size=n_data // num_users, random_state=42)

        train_datasets = {}
        test_datasets  = {}
        template = torchvision.datasets.SVHN(root=self.dataset_dir, 
                                            split='test', 
                                            download=False, 
                                            transform=self.transform)
        template.data = None
        template.labels = None

        for i, (data_indices, _) in enumerate(spliter_fn.split(self.data, self.labels)):
            if client_indices is not None:
                data_indices = client_indices[i]
            train_data, train_labels = self.data[data_indices], self.labels[data_indices]

            if client_indices is None:
                stratify = train_labels
            else:
                stratify = None

            X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42, stratify=stratify)
            train_datasets[i] = copy.deepcopy(template)
            train_datasets[i].data = X_train
            train_datasets[i].labels = Y_train
            
            test_datasets[i]  = copy.deepcopy(template)
            test_datasets[i].data = X_test
            test_datasets[i].labels = Y_test

        print('Done loading SVHN')
        return (train_datasets, test_datasets), 10

    def __load_dataset(self):
        train_dataset = torchvision.datasets.SVHN(root=self.dataset_dir, 
                                                   split='train', 
                                                   download=True, 
                                                   transform=self.transform)
        test_dataset  = torchvision.datasets.SVHN(root=self.dataset_dir, 
                                                   split='test', 
                                                   download=True, 
                                                   transform=self.transform)

        # self.data  = torch.concat((train_dataset.data, test_dataset.data), dim=0)
        # self.labels = torch.concat((train_dataset.targets, test_dataset.targets), dim=0)
        # self.labels = torch.unsqueeze(self.labels, 1)
        self.data  = np.concatenate((train_dataset.data, test_dataset.data), axis=0)
        self.labels = np.concatenate((train_dataset.labels, test_dataset.labels), axis=0)
        self.labels = np.expand_dims(self.labels, axis=1)


if __name__ == '__main__':
    dataManager = SVHN_DataManager()
    dataManager.get_datasets(50)