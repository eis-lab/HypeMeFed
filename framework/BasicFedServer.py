import copy
import torch

from .FedClients.FedClient import FedClient
from typing import List


class BasicFedServer:
    AGGREGATION_TYPE = ["fedavg", "hypernet", "exit_levelwise"]

    def __init__(self, aggregation_scheme='fedavg', r=100):
        if aggregation_scheme not in BasicFedServer.AGGREGATION_TYPE:
            raise ValueError("Invalid aggregation scheme: {}".format(aggregation_scheme))

        self.aggregation_scheme = aggregation_scheme
        self.round = 0
        self.r = r

    def aggregate(self, clients: List[FedClient], **kwargs):
        client_params = []
        self.round += 1
        for client in clients:
            client_params.append(client.get_fl_parameters())

        if self.aggregation_scheme == "fedavg":
            return self.aggregate_fedavg(client_params)
        elif self.aggregation_scheme == "hypernet":
            return self.aggregate_hypernet(client_params)
        elif self.aggregation_scheme == "exit_levelwise":
            return self.aggregate_exit_levelwise(client_params)
        else:
            raise ValueError("Invalid aggregation scheme: {}".format(self.aggregation_scheme))

    def aggregate_hypernet(self, client_models):
        model_name = str(type(client_models[0])).lower()

        if 'vgg' in model_name:
            from .Hypernetworks.HyperVGG import aggregate_hypernet as aggregate_hypernet_vgg
            client_models = aggregate_hypernet_vgg(self, client_models, self.round)
        
        aggregated_model = self.aggregate_fedavg(client_models)
        aggregated_models = {}

        for model in client_models:
            aggregated_models[model.exit_level] = copy.deepcopy(aggregated_model)

        return aggregated_models
    
    def aggregate_fedavg(self, client_models):
        """
        Aggregate client models using FedAvg.
        """
        aggregated_model = copy.deepcopy(client_models[0]) 

        for key in aggregated_model.state_dict().keys():
            for client_model in client_models[1:]:
                aggregated_model.state_dict()[key] += client_model.state_dict()[key]
            
            if aggregated_model.state_dict()[key].dtype == torch.float32:
                aggregated_model.state_dict()[key] /= len(client_models)
            elif aggregated_model.state_dict()[key].dtype == torch.int64:
                aggregated_model.state_dict()[key] //= len(client_models)
            else:
                aggregated_model.state_dict()[key] /= len(client_models)

        for model in client_models:
            model.load_state_dict(aggregated_model.state_dict())

        return aggregated_model

    def aggregate_exit_levelwise(self, client_models):
        aggregated_models = {}
        base_idxs = {}
        max_level = 0
        for i, model in enumerate(client_models):
            if model.exit_level not in aggregated_models:
                aggregated_models[model.exit_level] = copy.deepcopy(model)
                base_idxs[model.exit_level] = i
                
                if model.exit_level > max_level:
                    max_level = model.exit_level
        
        for key in aggregated_models[max_level].state_dict().keys():
            for level in range(1, max_level+1):
                if level not in aggregated_models:
                    continue
                
                target_model = aggregated_models[level]
                n_aggregated = 0
                for i, client_model in enumerate(client_models):
                    if base_idxs[level] == i:
                        continue
                    
                    if key in client_model.state_dict() and key in target_model.state_dict():
                        target_model.state_dict()[key] += client_model.state_dict()[key]
                        n_aggregated += 1
                
                if n_aggregated == 0:
                    continue
                
                if target_model.state_dict()[key].dtype == torch.float32:
                    target_model.state_dict()[key] /= n_aggregated
                elif target_model.state_dict()[key].dtype == torch.int64:
                    target_model.state_dict()[key] //= n_aggregated
                else:
                    target_model.state_dict()[key] /= n_aggregated
    
        return aggregated_models