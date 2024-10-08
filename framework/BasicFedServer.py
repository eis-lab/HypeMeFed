import copy
import torch

from .FedClients.FedClient import FedClient
from typing import List


class BasicFedServer:
    AGGREGATION_TYPE = ["fedavg", "hypernet"]

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
