import torch
import torch.nn as nn


class ACNetwork(nn.Module):
    """Modified from MinAtar example: 
      - https://github.com/kenjyoung/MinAtar/blob/master/examples/AC_lambda.py
    """
    def __init__(self, in_channels, num_actions, env_name):
        super(ACNetwork, self).__init__()
        
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        self.policy = nn.Linear(in_features=128, out_features=num_actions)
        self.value = nn.Linear(in_features=128, out_features=1)

    # As per implementation instructions, the forward function should be overwritten by all subclasses
    def forward(self, x):
        dSiLU = lambda x: torch.sigmoid(x) * (1 + x * (1 - torch.sigmoid(x)))
        SiLU = lambda x: x * torch.sigmoid(x)
        
        x = x.reshape((x.shape[0], -1, 10, 10))  # (n_samples, channels in env, 10, 10)
        x = SiLU(self.conv(x))
        x = dSiLU(self.fc_hidden(x.view(x.size(0), -1)))
        return self.policy(x), self.value(x)
