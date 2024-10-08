import torch
import torch.nn as nn


def decompose_param(conv_weight, r=64):
    in_channels, out_channels, kernel_size, _ = conv_weight.shape

    r = min(r, in_channels*kernel_size, out_channels*kernel_size)

    origianl_weight = conv_weight.view(in_channels*kernel_size, out_channels*kernel_size)

    U, S, Vh = torch.svd(origianl_weight)

    U_r = U[:, :r] 
    S_r = torch.diag(S[:r])
    V_r = Vh[:, :r]
    
    loraA = U_r @ torch.sqrt(S_r) # (in_channels*kernel_size, r)
    loraB = torch.sqrt(S_r) @ V_r.t() # (r, out_channels*kernel_size)

    return loraA.transpose(0, 1), loraB


def to_lora_params(conv_weight, r, init_with_svd=False):
    in_channels, out_channels, kernel_size, _ = conv_weight.shape

    loraA = nn.Parameter(conv_weight.new_zeros((r, in_channels*kernel_size)))
    loraB = nn.Parameter(conv_weight.new_zeros((r, out_channels*kernel_size)))

    if init_with_svd:
        svdA, svdB = decompose_param(conv_weight, r=r)
        loraA.data = svdA
        loraB.data = svdB

    return loraA, loraB


class LRFConvHyperNetwork(torch.nn.Module):
    def __init__(self, conv1, conv2, r=64, hidden_size=None, init_with_svd=True):
        super(LRFConvHyperNetwork, self).__init__()
        self.r = r
        self.target_shape = conv2.weight.shape
        self.init_with_svd = init_with_svd

        target_in_channels = conv2.in_channels
        target_out_channels = conv2.out_channels
        target_kernel_size = conv2.kernel_size[0]
        
        self.loraA_conv1, self.loraB_conv1 = to_lora_params(conv1.weight.data, r=r, init_with_svd=init_with_svd)

        if hidden_size:
            self.loraA_mapper = nn.Sequential(
                nn.Linear(self.loraA_conv1.shape[1], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, target_in_channels*target_kernel_size)
            ) 

            self.loraB_mapper = nn.Sequential(
                nn.Linear(self.loraB_conv1.shape[1], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, target_out_channels*target_kernel_size)
            )
        else:
            self.loraA_mapper = nn.Linear(self.loraA_conv1.shape[1], target_in_channels*target_kernel_size)
            self.loraB_mapper = nn.Linear(self.loraB_conv1.shape[1], target_out_channels*target_kernel_size)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(0)
        loraA_conv1, loraB_conv1 = decompose_param(x, r=self.r)

        loraA_mapped = self.loraA_mapper(loraA_conv1)
        loraB_mapped = self.loraB_mapper(loraB_conv1)

        x = loraB_mapped.transpose(0, 1) @ loraA_mapped 
        x = x.view(self.target_shape)
        x.unsqueeze(0)

        return x
 
 