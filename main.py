import os
from typing import List
import numpy as np
import torch
import torch.utils.tensorboard as tb
import argparse
import ast
import natsort
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import copy

from framework.BasicFedServer import BasicFedServer
from framework.FedClients.FedClient import FedClient
from framework.DataManager import get_datasets
from framework.FedModelGenerator import FedModelGenerator

def args_as_list(args):
    v = ast.literal_eval(args)
    if type(v) is not list:
        raise argparse.ArgumentTypeError(f"Argument {args} must be a list")
    return v


def select_participants(clients, num_participants, round, n_exit_levels):
    if round >= 0:
        num_parts = [num_participants // n_exit_levels] * n_exit_levels
        if sum(num_parts) < num_participants:
            num_parts[1] += num_participants - sum(num_parts)
        
        ## default
        participating_client_idxs = list(np.random.choice(range(0, NUM_OF_CLIENTS, n_exit_levels), num_parts[0], replace=False))
        if n_exit_levels > 1:
            for i in range(1, len(EXIT_LEVELS)):
                participating_client_idxs += list(np.random.choice(range(i, NUM_OF_CLIENTS, n_exit_levels), num_parts[i], replace=False))

        return participating_client_idxs
    
    largest_ratio = 0.6

    exit_level_models = {
        i: [] for i in range(n_exit_levels)
    }

    for i, client in enumerate(clients):
        exit_level_models[client.global_model.exit_level - 1].append(i)
    
    participating_client_idxs = []
    for i in range(n_exit_levels):
        if i == n_exit_levels - 1:
            _num_participants = int(num_participants * largest_ratio)
            participating_client_idxs += list(np.random.choice(exit_level_models[i], _num_participants, replace=False))
        else:
            _num_participants = int(num_participants * (1 - largest_ratio) / (n_exit_levels - 1))
            participating_client_idxs += list(np.random.choice(exit_level_models[i], _num_participants, replace=False))
    
    return participating_client_idxs


db_hyperparams = {
    'unimib': {
        "lr": 0.005,
        'batch_size': 16
    },
    'stl10': {
        'lr': 0.0005,
        'batch_size': 32
    },
    'svhn': {
        'lr': 0.0005,
        'batch_size': 128
    },
}

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='unimib', choices=['stl10', 'svhn', 'unimib'])
    parser.add_argument('--model', type=str, default='vgg', choices=['vgg'])
    parser.add_argument('--multi_exit', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--use_hn', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--scheme', type=str, default='joint', choices=['joint'])
    parser.add_argument('--exit_levels', type=args_as_list, default=[1,2,3])
    parser.add_argument('--iid', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_participants', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=300)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--n_exits', type=int, default=3)
    parser.add_argument('--tag', type=str, default='')

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    OPTIMIZER = 'adam' #'adam' / 'sgd'

    BATCH_SIZE    = args.batch_size
    LEARNING_RATE = args.lr

    if args.dataset in db_hyperparams:
        LEARNING_RATE = db_hyperparams[args.dataset]['lr']
        BATCH_SIZE    = db_hyperparams[args.dataset]['batch_size']

    PARTICIPANTS         = args.num_participants
    NUM_OF_CLIENTS       = args.num_clients
    LOCAL_BATCH_SIZE     = BATCH_SIZE
    LOCAL_EPOCH          = args.local_epoch
    ROUNDS               = args.rounds

    if args.dataset == 'unimib':
        print("Unimib detected")
        PARTICIPANTS = 6
        NUM_OF_CLIENTS = 30

    GPU_NUM = args.gpu
    IS_CUDA = torch.cuda.is_available()
    DEVICE  = torch.device('cuda:' + str(GPU_NUM) if IS_CUDA else 'cpu')

    EXIT_LEVELS       = args.exit_levels
    USE_HYPERNET      = args.use_hn == 'True'
    if len(EXIT_LEVELS) == 1:
        USE_HYPERNET = False

    MULTIEXIT_SCHEME  = args.scheme
    EXIT_LEVELS_STR   = f'[{",".join(map(str, EXIT_LEVELS))}]'
    IS_MULTI_EXIT     = args.multi_exit == 'True'

    print(EXIT_LEVELS)
    print(DEVICE)

    TAG    = args.tag
    TAG    += f'_{MULTIEXIT_SCHEME}'
    TAG    += f"_noniid{args.iid}" if args.iid >= 0.01 else ""
    TAG    += f"_hypernet" if USE_HYPERNET else ""

    CONFIG_NAME = f'HN_{args.dataset.upper()}_MultiExit_{args.multi_exit}_model-{args.model}' + \
                  f'_rounds{args.rounds}' + \
                  f'_NC{NUM_OF_CLIENTS}_NP{PARTICIPANTS}_LR{LEARNING_RATE}_seed{args.seed}' + \
                  f'_bs{BATCH_SIZE}_Levels={EXIT_LEVELS_STR}_{TAG}'

    print(CONFIG_NAME)

    (train_datasets, test_datasets), NUM_CLASSES = get_datasets(args.dataset, NUM_OF_CLIENTS, args.iid)

    global_model_name = args.model if args.multi_exit == 'False' else 'multiexit_' + args.model
    global_model_params = {'model_name': global_model_name, 
                           'dataset_name': args.dataset,
                           'model_param': {'num_classes': NUM_CLASSES}}
    global_model_params["model_param"]["exit_level"] = 3

    global_model = FedModelGenerator.generate_from_name_for_dataset(global_model_params['dataset_name'],
                                                             global_model_params['model_name'],
                                                             None if 'model_param' not in global_model_params 
                                                                 else global_model_params['model_param']).to(DEVICE)
    
    clients: List[FedClient] = []
    for user_idx in range(NUM_OF_CLIENTS):
        train_data, test_data = train_datasets[user_idx], test_datasets[user_idx]

        model = copy.deepcopy(global_model).to(DEVICE)
        exit_level = EXIT_LEVELS[user_idx%len(EXIT_LEVELS)]
        if IS_MULTI_EXIT:
            model.exit_level = exit_level
            global_model_params["model_param"]["exit_level"] = exit_level

        multiexit_params = {
                'num_classes': NUM_CLASSES,
                'scheme': MULTIEXIT_SCHEME
        } if IS_MULTI_EXIT else None

        clients.append(FedClient(global_model=model,
                                       global_model_params=global_model_params,
                                       multiexit_params=multiexit_params,
                                       train_dataset=train_data,
                                       test_dataset=test_data,
                                       batch_size=LOCAL_BATCH_SIZE,
                                       optimizer=OPTIMIZER, 
                                       learning_rate=LEARNING_RATE, 
                                       local_epoch=LOCAL_EPOCH, 
                                       device=DEVICE, 
                                       print_logs=True))
    
    if len(EXIT_LEVELS) == 1:
        scheme = 'fedavg'
    else:
        scheme = 'exit_levelwise'

    if USE_HYPERNET:
        scheme = 'hypernet'
        print(scheme)

    fed_server = BasicFedServer(aggregation_scheme=scheme)

    loaded_round = -1

    for round in range(loaded_round+1, ROUNDS):
        print(CONFIG_NAME)
        print("Round: ", round)

        for client in clients:
            client.set_global_round(round +1)
        
        accs_by_exit_level_global = {i: [] for i in range(0, args.n_exits+1)} # 0 for last exit
        accs_by_exit_level_local = {i: [] for i in range(0, args.n_exits+1)} # 0 for last exit

        n_exit_levels = len(EXIT_LEVELS)
        participating_client_idxs = select_participants(clients, PARTICIPANTS, round, n_exit_levels)
        participating_clients = [clients[i] for i in participating_client_idxs]
        
        losses_by_model_exit_level = {}
        for i, client in enumerate(participating_clients):
            loss = client.train()

            model_level = len(loss)
            if model_level not in losses_by_model_exit_level:
                losses_by_model_exit_level[model_level] = loss
            else:
                for idx, (cnt_loss, dist_loss) in enumerate(loss):
                    losses_by_model_exit_level[model_level][idx][0] += cnt_loss
                    losses_by_model_exit_level[model_level][idx][1] += dist_loss

            accs_local, accs_global = client.test() # (num_exits, )
            
            for exit_level, acc in enumerate(accs_global):
                accs_by_exit_level_global[exit_level+1].append(acc)
            for exit_level, acc in enumerate(accs_local):
                accs_by_exit_level_local[exit_level+1].append(acc)
            accs_by_exit_level_global[0].append(accs_global[-1])
            accs_by_exit_level_local[0].append(accs_local[-1])
            
            print(f"Client {i}_Accuracy: {accs_global[-1]*100:.2f}%")
            print()

        acc_for_exit_level_global = {}
        acc_for_exit_level_local = {}

        for exit_level, accs in accs_by_exit_level_global.items():
            acc_for_exit_level_global[exit_level] = np.mean(accs) 
        for exit_level, accs in accs_by_exit_level_local.items():
            acc_for_exit_level_local[exit_level] = np.mean(accs)

        print(f"Aggregating..")
        aggregatation_result = fed_server.aggregate(participating_clients)

        if scheme == 'fedavg':
            for client in clients:
                client.global_model.load_state_dict(aggregatation_result.state_dict())
        else:
            for client in clients:
                exit_level = client.global_model.exit_level
                if exit_level not in aggregatation_result:
                    continue
                client.global_model.load_state_dict(aggregatation_result[exit_level].state_dict())

        acc_for_exit_level_general_model = {}
        _acc_for_exit_level_general_model = {i: [] for i in range(0, args.n_exits+1)}
        for client in clients:
            accs_local, accs_global = client.test()
            for exit_level, acc in enumerate(accs_global):
                _acc_for_exit_level_general_model[exit_level+1].append(acc)
            _acc_for_exit_level_general_model[0].append(accs_global[-1])
        
        for exit_level, accs in _acc_for_exit_level_general_model.items():
            acc_for_exit_level_general_model[exit_level] = np.mean(accs)

        print(f"Round {round} - Average Accuracy: Local ({acc_for_exit_level_global[0]*100:.2f}) | General ({acc_for_exit_level_general_model[0]*100:.2f})")

        if args.multi_exit == 'True':
            for exit_level in range(1, args.n_exits+1):
                print(f"Exit(Global) {exit_level}: {acc_for_exit_level_global[exit_level] * 100:.2f}%", end=' ')
                print(f"Exit(General) {exit_level}: {acc_for_exit_level_general_model[exit_level] * 100:.2f}%", end=' ')
            print()

