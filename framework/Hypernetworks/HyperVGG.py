import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp


from torch.utils.data import DataLoader
from framework.Hypernetworks.LRFConvHyperNetwork import *
from framework.Hypernetworks.WeightDataset import WeightDataset

class NaiveHyperNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, target_layer=None):
        super(NaiveHyperNetwork, self).__init__()

        self.target_shape = None
        if target_layer:
            self.target_shape = target_layer.weight.data.shape

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.contiguous().view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        if self.target_shape:
            x = x.view(self.target_shape)

        return x


def train_hypernetwork(hypernet, optimizer, criterion, train_loader, device, num_epochs=100):
    hypernet.train()
    hypernet.to(device)
    criterion.to(device)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            source_batch, target_batch = batch
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            
            output = hypernet(source_batch)
            output = output.squeeze(0)
            target_batch = target_batch.squeeze(0)

            loss = criterion(output, target_batch)
    
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


def evaluate_hypernetwork(hypernet, input_weights, device):
    hypernet.eval()
    outputs = []
    with torch.no_grad():
        for input_weight in input_weights:
            input_weight = input_weight.to(device)
            input_weight = input_weight.unsqueeze(0)

            output = hypernet(input_weight)
            output = output.squeeze(0)
            outputs.append(output)

    return outputs


def initialize_hypernetworks(server, model, device, r):
    if not hasattr(server, 'hypernets'):
        server.source_target_dict = {}

        named_moduels = dict(model.named_modules())
        for i in range(1, 3):
            target_exit_levels = list(range(i+1, 4))
            for conv_idx in [0, 3, 6]:
                source_name = f'exit{i}.layers.{conv_idx}'

                if source_name not in named_moduels:
                    continue

                source_conv = named_moduels[source_name]

                if not isinstance(source_conv, nn.Conv2d):
                    continue

                source = (source_name, source_conv)

                target_name = f'exit{i+1}.layers.{conv_idx}'
                target_conv = named_moduels[target_name]
                target = (target_name, target_conv)
                server.source_target_dict[(source, target)] = target_exit_levels

        server.hypernets = {}
        
        for source, target in server.source_target_dict.keys():
            source_name, source_conv = source
            target_name, target_conv = target
            
            if ('layers.1' in source_name) or ('layers.4' in source_name):
                hypernet = NaiveHyperNetwork(source_conv.weight.size(0), target_conv.weight.size(0), hidden_size=64).to(device)
            else:
                hypernet = LRFConvHyperNetwork(source_conv, 
                                               target_conv, r=r, hidden_size=512, init_with_svd=True).to(device)
                
            server.hypernets[(source_name, target_name)] = hypernet
        
    return server.source_target_dict, server.hypernets

def aggregate_hypernet(server, client_models, r=100):
    device = next(client_models[0].parameters()).device
    exit_level3_model = [model for model in client_models if model.exit_level == 3][0]
    source_target_dict, hypernets = initialize_hypernetworks(server, exit_level3_model, device, r=r)
    lr = 0.0005

    optimizers = {source_target_name_pair: optim.Adam(hypernet.parameters(), lr=lr) 
                  for source_target_name_pair, hypernet in hypernets.items()}
    criterion = {name: nn.MSELoss().to(device) for name in hypernets.keys()}

    level1_models = [model for model in client_models if model.exit_level == 1]
    level2_models = [model for model in client_models if model.exit_level == 2]
    level3_models = [model for model in client_models if model.exit_level == 3]

    datasets = {}
    target_exits = {}

    for (source, target), target_exit_levels in source_target_dict.items():
        source_name, _ = source
        target_name, _ = target

        if len(target_exit_levels) == 1:
            target_models = level3_models
        elif len(target_exit_levels) == 2:
            target_models = level2_models + level3_models

        train_source_weights = [model.state_dict()[source_name+'.weight'].data for model in target_models]
        train_target_weights = [model.state_dict()[target_name+'.weight'].data for model in target_models]

        datasets[(source_name, target_name)] = WeightDataset(train_source_weights, train_target_weights)
        target_exits[source_name] = target_exit_levels

    num_epochs = 25

    for hypernet_name, hypernet in hypernets.items():
        dataloader = DataLoader(datasets[hypernet_name], batch_size=1, shuffle=True)
        train_hypernetwork(hypernet, optimizers[hypernet_name], criterion[hypernet_name], 
                           dataloader, device, num_epochs=num_epochs)

    for hypernet_name, hypernet in hypernets.items():
        source_name, target_name = hypernet_name

        if len(target_exits[source_name]) == 1:
            target_models = level1_models + level2_models
        elif len(target_exits[source_name]) == 2:
            target_models = level1_models

        for model in target_models:
            weights = [model.state_dict()[source_name+'.weight'].data]
            generated = evaluate_hypernetwork(hypernet, weights, device)
            model.state_dict()[target_name+'.weight'].data = generated[0]            

    return client_models

