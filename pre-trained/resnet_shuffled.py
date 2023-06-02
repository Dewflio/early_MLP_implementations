import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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
    

class ShuffledConv2dVariable(nn.modules.Conv2d):
    def __init__(self, in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True,
        padding_mode = 'zeros',
        device=None,
        dtype=None, 
        n_shuffled=None):
        super(ShuffledConv2dVariable, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device= device,
            dtype= dtype)
        
        self.n_shuffled = n_shuffled if (n_shuffled is not None and n_shuffled < self.out_channels) else self.out_channels

    def change_n_shuffled(self, n_shuffled: int):
        self.n_shuffled = n_shuffled if (n_shuffled is not None and n_shuffled < self.out_channels) else self.out_channels
        return self.n_shuffled
    
    def forward(self, input: Tensor) -> Tensor:
        indices = torch.argsort(torch.rand_like(self.weight.T), dim=-1)

        print(indices)
        if self.n_shuffled < self.out_channels:
            rows_to_sort = random.sample(range(self.out_channels), (self.out_channels - self.n_shuffled))
            sorted_rows, _ = torch.sort(indices[:, :, :, rows_to_sort], dim=-1)
            indices[:, :, :, rows_to_sort] = sorted_rows

        print(indices)
        shuffled_W_T = torch.gather(self.weight.T, dim=-1, index=indices)
        shuffled_W = shuffled_W_T.T
        return self._conv_forward(input, shuffled_W, self.bias)
    
class ShuffledResnetVariableConv2d(nn.Module):
    def __init__(self, model):
        super(ShuffledResnetVariableConv2d, self).__init__()
        self.resnet_model = model
        self.layers = list(self.resnet_model.children())
        self.modified_layer = ShuffledConv2dVariable(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False, n_shuffled=0)
        self.modified_layer.weight = self.layers[0].weight
        self.modified_layer.bias = self.layers[0].bias
        self.layers[0] = self.modified_layer

    
    def change_n_shuffled(self, n_shuffled: int):
        return self.modified_layer.change_n_shuffled(n_shuffled=n_shuffled)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = torch.flatten(x, 1)
            x = layer(x)
        return x
    

if __name__ == "__main__":
    from torchvision.models import resnet50, ResNet50_Weights
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) 

    res = resnet50(weights = ResNet50_Weights.DEFAULT)
    
    dummy_tensor = torch.randn(1, 3, 227, 227)
    layer = ShuffledConv2dVariable(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False, n_shuffled=0)


    mod_model = ShuffledResnetVariableConv2d(res)
    mod_model.eval()

    mod_model.forward(dummy_tensor)
