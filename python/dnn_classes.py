import os
import sys
import timeit

import torch

class HiddenLayer(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(HiddenLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.linear = torch.nn.Linear(self.n_in, self.n_out, bias=True)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        h = self.linear(x)
        return self.activation(h)

