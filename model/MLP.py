import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden=1024, n_layers=2, drop_out=0.5, batchnorm=True):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, n_hidden))
        if batchnorm:
            self.layers.append(nn.BatchNorm1d(n_hidden))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(drop_out))
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
            if batchnorm:
                self.layers.append(nn.BatchNorm1d(n_hidden))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(drop_out))
        self.layers.append(nn.Linear(n_hidden, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

