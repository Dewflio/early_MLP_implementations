import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet18, ResNet18_Weights
import random

class ShuffledLinearVariable(nn.modules.Linear):
    def __init__(self, *args, n_shuffled=None):
        super(ShuffledLinearVariable, self).__init__(*args)
        self.n_shuffled = n_shuffled if (n_shuffled is not None and n_shuffled < self.in_features) else self.in_features

    def change_n_shuffled(self, n_shuffled: int):
        self.n_shuffled = n_shuffled if (n_shuffled is not None and n_shuffled < self.in_features) else self.in_features
        return self.n_shuffled
    
    def forward(self, input: Tensor) -> Tensor:
        indices = torch.argsort(torch.rand_like(self.weight.T), dim=-1)

        if self.n_shuffled < self.in_features:
            rows_to_sort = random.sample(range(self.in_features), (self.in_features - self.n_shuffled))
            sorted_rows, _ = torch.sort(indices[rows_to_sort], dim=1)
            indices[rows_to_sort] = sorted_rows

        shuffled_W_T = torch.gather(self.weight.T, dim=-1, index=indices)
        shuffled_W = shuffled_W_T.T
        return F.linear(input, shuffled_W, self.bias)
    
class ShuffledResnet18Variable(nn.Module):
    def __init__(self, weights=ResNet18_Weights.DEFAULT):
        super(ShuffledResnet18Variable, self).__init__()
        self.resnet_model = resnet18(weights=weights)
        self.layers = []
        self.modified_layer: ShuffledLinearVariable = None
        for i, child in enumerate(self.resnet_model.children()):
            if i == len(list(self.resnet_model.children())) - 1:
                modified_layer = ShuffledLinearVariable(512, 1000, True)
                modified_layer.weight = child.weight
                modified_layer.bias = child.bias
                self.layers.append(modified_layer)
                self.modified_layer = self.layers[-1]
            else:
                self.layers.append(child)
            pass
    
    def change_n_shuffled(self, n_shuffled: int):
        return self.modified_layer.change_n_shuffled(n_shuffled=n_shuffled)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = torch.flatten(x, 1)
            x = layer(x)
        return x
    

class ShuffledResnetVariable(nn.Module):
    def __init__(self, model):
        super(ShuffledResnetVariable, self).__init__()
        self.resnet_model = model
        self.layers = list(self.resnet_model.children())
        self.modified_layer = ShuffledLinearVariable(self.resnet_model.fc.in_features, self.resnet_model.fc.out_features, True)
        self.modified_layer.weight = self.resnet_model.fc.weight
        self.modified_layer.bias = self.resnet_model.fc.bias
        self.layers[-1] = self.modified_layer
    
    def change_n_shuffled(self, n_shuffled: int):
        return self.modified_layer.change_n_shuffled(n_shuffled=n_shuffled)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = torch.flatten(x, 1)
            x = layer(x)
        return x