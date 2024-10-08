import numpy as np


def iid_partition(dataset, n_clients):
    num_items_per_client = int(len(dataset) / n_clients)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]

    for i in range(n_clients):
        client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
        image_idxs = list(set(image_idxs) - client_dict[i])

    return client_dict

def non_iid_partition(labels, n_clients, alpha=0.5, min_data_size=200):
    y_train = np.array(labels)
    K = len(np.unique(y_train))
    N = y_train.shape[0]
    client_dict = {}
    min_size = 0

    while min_size < min_data_size:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, split_points))]

        sizes = [len(idx_j) for idx_j in idx_batch]
        min_size = min(sizes)

    for j in range(n_clients):
        np.random.shuffle(idx_batch[j])
        client_dict[j] = np.array(idx_batch[j])

    return client_dict


def get_datasets(dataset_name, num_clients, alpha=10000):
    if dataset_name == 'unimib':
        from utilities.unimib_dataset import UniMib_DataManager

        data_manager = UniMib_DataManager('datasets/Unimib_SHAR/data/')
        labels = data_manager.labels
        min_size = 24 if alpha > 0.15 else 13
        client_indices = non_iid_partition(labels, num_clients, alpha, min_size)
        return data_manager.get_datasets(3, 151, num_clients, client_indices=client_indices)
    elif dataset_name == 'stl10':
        from utilities.stl_dataset import STL10_DataManager

        data_manager = STL10_DataManager(dataset_dir='datasets/stl10/')
        labels = data_manager.labels
        client_indices = non_iid_partition(labels, num_clients, alpha, min_data_size=50)
        return data_manager.get_datasets(num_clients, client_indices)
    elif dataset_name == 'svhn':
        from utilities.svhn_dataset import SVHN_DataManager
        
        data_manager = SVHN_DataManager(dataset_dir='datasets/svhn/')
        labels = data_manager.labels
        client_indices = non_iid_partition(labels, num_clients, alpha, min_data_size=50)
        return data_manager.get_datasets(num_clients, client_indices)
    else:
        raise ValueError("Invalid dataset name: {}".format(dataset_name))    
