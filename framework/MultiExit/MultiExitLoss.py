import torch.nn as nn
import torch.nn.functional as F


class DistillationBasedLoss(nn.Module):
    def __init__(self, C=0, maxprob=0.5, Tmult=1.05, global_scale=1.0):
        super(DistillationBasedLoss, self).__init__()
        self.C = C
        self.maxprob = maxprob
        self.Tmult = Tmult
        self.T = 1.0
        self.global_scale = global_scale
        self.loss_per_exit = []
        self.loss_list = []


    def forward(self, multi_exit_preds, y):
        cum_loss = F.cross_entropy(multi_exit_preds[-1], y)
        self.loss_list = [cum_loss]

        self.loss_per_exit = [(cum_loss.item(), 0)]
        if len(multi_exit_preds) == 1:
            return cum_loss
        
        prob_t = F.softmax(multi_exit_preds[-1].data / self.T, dim=1)
        for logits in multi_exit_preds[:-1]:
            logprob_s = F.log_softmax(logits / self.T, dim=1)
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss
            self.loss_list.append(cross_ent)
            self.loss_per_exit.append((cross_ent.item(), dist_loss.item()))

        self.loss_per_exit.reverse()
        self.loss_list.reverse()
        
        adj_maxprob = prob_t.max(dim=1)[0].mean()
        if adj_maxprob > self.maxprob:
            self.T *= self.Tmult

        return cum_loss * self.global_scale


