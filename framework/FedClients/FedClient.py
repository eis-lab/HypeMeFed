import torch

from torch.utils.data import DataLoader
from framework.MultiExit.MultiExitLoss import DistillationBasedLoss

class FedClient:
    def __init__(self, 
                 global_model,
                 global_model_params: dict,
                 batch_size,
                 train_dataset, 
                 test_dataset,
                 optimizer,
                 learning_rate,
                 local_epoch,
                 device,
                 multiexit_params:dict={},
                 print_logs=False):
        self.global_model = global_model
        self.num_classes = global_model_params['model_param']['num_classes']

        self.batch_size    = batch_size if len(train_dataset) > batch_size else len(train_dataset)
        self.train_loader  = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_loader   = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=True)
        self.print_logs    = print_logs

        self.multiexit     = False if multiexit_params == {} or multiexit_params is None else True
        self.basic         = not self.multiexit
        self.dataset_name  = global_model_params['dataset_name']

        if self.multiexit:
            self.multiexit_scheme = multiexit_params['scheme'] if 'scheme' in multiexit_params else 'joint'
            if self.multiexit_scheme == 'joint':
                    self.criterion = DistillationBasedLoss().to(device)
            else:
                self.criterion = torch.nn.CrossEntropyLoss().to(device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to(device)

        self.optimizer     = optimizer
        self.learning_rate = learning_rate
        self.local_epoch   = local_epoch
        self.device        = device
        self.round         = 0

    def set_global_round(self, round):
        self.round = round

    def get_fl_parameters(self):
        return self.global_model

    def test(self, **kwargs):
        if self.basic:
            acc = self._test(self.global_model,
                             self.test_loader,
                             self.device,
                             **kwargs)
            
            return [acc], [acc]

        if self.multiexit:
            # Legacy code for mutual learning
            acc_global = self._test_multiexit(self.global_model,
                                              self.test_loader,
                                              self.device,
                                              **kwargs)
            return acc_global, acc_global
        else:
            acc_global = self._test(self.global_model,
                                self.test_loader,
                                self.device,
                                **kwargs)
            return [acc_global], [acc_global]


    def train(self, **kwargs):
        if self.optimizer  == 'adam':
            self.optimizer = torch.optim.Adam(self.global_model.parameters(), 
                                              lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.global_model.parameters(), 
                                             lr=self.learning_rate, 
                                             momentum=0.5, 
                                             weight_decay=0.0001)

        if self.basic:
            print('train basic')
            losses = self._train_basic(self.global_model,
                                       self.train_loader,
                                       self.criterion,
                                       self.learning_rate,
                                       self.local_epoch,
                                       self.device,
                                       global_optimizer=self.optimizer,
                                       print_logs=self.print_logs,
                                       **kwargs)
            return losses    
        

        train_schemes = {
            'joint': self._train_multiexit,
        }
        train_method = train_schemes[self.multiexit_scheme]
        losses = train_method(self.global_model,
                                self.train_loader,
                                self.criterion,
                                self.learning_rate,
                                self.local_epoch,
                                self.device)
        
        return losses


    def _train_multiexit(self,
                         model,
                         train_loader,
                         criterion,
                         lr,
                         epochs,
                         device,
                         optimizer=None):
        model.to(device)
        model.train()

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr*0.1)

        i = 1
        losses = None
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                data, target = batch[0].to(device), batch[1].to(device)
                target = target.squeeze()
                preds, _ = model(data)

                loss = criterion(preds, target)
                loss.backward()
                optimizer.step()
                losses_global = criterion.loss_per_exit

                if losses is None:
                    losses = []
                    for i in range(len(losses_global)):
                        losses.append([losses_global[i][0], losses_global[i][1]])
                else:
                    for i in range(len(losses_global)):
                        losses[i] = [losses[i][0] + losses_global[i][0], losses[i][1] + losses_global[i][1]]

                i += 1
        
        for i in range(len(losses)):
            losses[i] = [losses[i][0] / len(train_loader), losses[i][1] / len(train_loader)]
        return losses

    def _test_multiexit(self,
                        model,
                        test_loader, 
                        device):
        model.to(device)
        model.eval()
        corrects = None
        with torch.no_grad():
            for batch in test_loader:
                data, target = batch[0].to(device), batch[1].to(device)
                target = target.squeeze()
                multi_exit_preds, _ = model(data)

                # multi_exit_preds, pred = model(data) # B x NUM_EXITS x NUM_CLASSES, B x NUM_CLASSES
                multi_exit_preds = torch.stack(multi_exit_preds, dim=1).to(device) 
                prediction_by_exit = torch.argmax(multi_exit_preds, dim=2) # B x NUM_EXITS
                num_corrects_by_exit = torch.sum(prediction_by_exit == target.unsqueeze(1), dim=0) # NUM_EXITS
                if corrects is None:
                    corrects = num_corrects_by_exit
                else:
                    corrects += num_corrects_by_exit

        
        acc_tests = corrects.float() / len(test_loader.dataset)

        return acc_tests.cpu().numpy()
    

    def _train_basic(self,
               model,
               train_loader, 
               criterion, 
               lr, 
               epochs, 
               device, 
               global_optimizer=None):
        model.to(device)
        model.train()

        if global_optimizer is None:
            global_optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        i = 1
        losses = []

        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_loader):
                # data, target = data.to(device), target.to(device)
                global_optimizer.zero_grad()

                data, target = batch[0].to(device), batch[1].to(device)
                target = target.squeeze()
                output = model(data)
                
                loss = criterion(output, target)

                loss.backward()

                losses.append(loss.item())
                global_optimizer.step()

                i += 1
        
        return [[sum(losses) / len(losses)] * 2]
    
    def _test(self,
              model,
              test_loader, device):
        model.to(device)
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in test_loader:

                data, target = batch[0].to(device), batch[1].to(device)
                target = target.squeeze()
                output = model(data)

                prediction = output.data.max(1)[1]
                correct += prediction.eq(target.data).sum()
        
        acc_test = correct / len(test_loader.dataset)

        return acc_test.item()