from torch.utils.data import Dataset

class WeightDataset(Dataset):
    def __init__(self, source_weights, target_weights, flatten_source=False, flatten_target=False):
        self.source_weights = source_weights
        self.target_weights = target_weights
    
        if flatten_source:
            self.source_weights = [weight.view(-1) for weight in self.source_weights]
        if flatten_target:
            self.target_weights = [weight.view(-1) for weight in self.target_weights]

    def __len__(self):
        return len(self.source_weights)
    
    def __getitem__(self, idx):
        return self.source_weights[idx], self.target_weights[idx]
