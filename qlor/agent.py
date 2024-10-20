from torch import nn
import torch
import numpy as np


class Agent(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(Agent, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        conv_out_size = self._get_conv_out(input_shape)
        avg_pool_out_size = self._get_avg_pool_out(conv_out_size)
        flatten_out_size = self._get_flatten_out(avg_pool_out_size)    
    
        self.fc = nn.Sequential(
            nn.Linear(flatten_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return o.size()[1:]

    def _get_avg_pool_out(self, shape):
        o = self.avg_pool(torch.zeros(1, *shape))
        return o.size()[1:]

    def _get_flatten_out(self, shape):
        o = self.flatten(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
