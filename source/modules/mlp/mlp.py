import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self,
                 layers_dim: list,
                 function=torch.relu,
                 bias=False
                 ):
        super(MLP, self).__init__()
        self.layers_num = len(layers_dim)
        self.layers_dim = layers_dim
        self.bias = bias
        self.function = function

        self.layers = nn.ModuleList([nn.Linear(in_features=x, out_features=y, bias=self.bias)
                                     for x, y in zip(self.layers_dim[:-1], self.layers_dim[1:])])

    def forward(self, inputs):

        out = inputs

        for cov in self.layers[:-1]:
            out = self.function(cov(out))

        out = self.layers[-1](out)

        return out


