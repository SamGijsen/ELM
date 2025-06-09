import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel

from torch import Tensor

class SSL_Projector(BaseModel):
    def __init__(self, in_dim=64, dim=128, out_dim=64, n_layers=3, bn=False):
        super().__init__()
        # we need an initial and final layer.
        assert n_layers > 1

        # first layer
        layers = [nn.Linear(in_dim, dim)]
        if bn:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.ReLU(inplace=True))

        # append additional middle layers depending on n_layers
        for _ in range(n_layers - 2):  # Subtract 2 to account for the first and last layers
            layers.append(nn.Linear(dim, dim))
            if bn:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))

        # last layer
        layers.append(nn.Linear(dim, out_dim, bias=False))
        # layers += [
        #     nn.Linear(dim, out_dim)
        #     #nn.ReLU(inplace=True)
        # ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class EEG_ResNet(BaseModel):
    def __init__(self, 
                 in_channels=1, 
                 conv1_params=[(4, 32, 1), (8, 32, 1), (16, 32, 1)], # size, dim, stride
                 n_blocks=4, 
                 res_params=[(4, 32, 1), (8, 32, 1), (16, 32, 1)],
                 res_pool_size=[4, 4, 4, 4],
                 dropout_p=False,
                 res_dropout_p=False,
                 proj_size=None): 
        super().__init__()
        self.dropout_p = dropout_p
        self.init_dim = sum([k[1] for k in conv1_params])
        self.res_dim = sum([k[1] for k in res_params])
        self.proj_size = proj_size
        assert self.init_dim == self.res_dim # currently only allow for differences in kernel size or stride

        # initial conv block
        self.conv1_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReflectionPad1d((kernel_size // 2, (kernel_size - 1) // 2)),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        )            
        for kernel_size, out_channels, stride in conv1_params])

        self.bn1 = nn.BatchNorm1d(self.init_dim)
        self.elu1 = nn.ELU(inplace=True)
        
        # residual blocks
        blocks = [EEG_ResNet_ResidualBlock(self.res_dim, res_params, res_pool_size[i], res_dropout_p=res_dropout_p)
                  for i in range(n_blocks)]
        self.res_blocks = nn.Sequential(*blocks)

        # post res blocks
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        if self.dropout_p:
            self.dropout = nn.Dropout(self.dropout_p)
        self.flat = nn.Flatten(start_dim=1)

        if self.proj_size:
            if len(self.proj_size) == 2:
                self.proj = nn.Sequential(
                    nn.Linear(self.res_dim, self.proj_size[0]),
                    nn.BatchNorm1d(self.proj_size[0]),
                    nn.ELU(inplace=True),
                    nn.Linear(self.proj_size[0], self.proj_size[1], bias=False))
            elif len(self.proj_size) == 1:
                self.proj = nn.Sequential(
                    nn.Linear(self.res_dim, self.proj_size[0], bias=False)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_outputs = [conv_layer(x) for conv_layer in self.conv1_layers]
        x = torch.cat(conv_outputs, dim=1)
        x = self.bn1(x)
        x = self.elu1(x)

        x = self.res_blocks(x)

        x = self.average_pool(x)
        if self.dropout_p:
            x = self.dropout(x)
        x = self.flat(x)

        if self.proj_size:
            proj_x = self.proj(x)
            return x, proj_x
        else:
            return x

class EEG_ResNet_ResidualBlock(BaseModel):
    def __init__(self, in_channels, 
                 conv_params=[(4, 32, 1), (8, 32, 1), (16, 32, 1)],
                 pool_size=4,
                 res_dropout_p=False):
        super(EEG_ResNet_ResidualBlock, self).__init__()
        self.pool_size = pool_size
        self.res_dropout_p = res_dropout_p

        # first conv with kernel_size and stride
        self.conv1_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReflectionPad1d((kernel_size // 2, (kernel_size - 1) // 2)),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        )            
        for kernel_size, out_channels, stride in conv_params])

        self.bn1 = nn.BatchNorm1d(sum([k[1] for k in conv_params]))
        self.elu1 = nn.ELU(inplace=True)

        # second conv with kernel_size and stride=1
        self.conv2_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReflectionPad1d((kernel_size // 2, (kernel_size - 1) // 2)),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1)
        )            
        for kernel_size, out_channels, stride in conv_params])

        self.bn2 = nn.BatchNorm1d(sum([k[1] for k in conv_params]))
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        # skip conv with kernel_size=1 and stride
        self.stream_conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
            for _, out_channels, stride in conv_params
        ])
        self.stream_bn = nn.BatchNorm1d(sum([k[1] for k in conv_params]))

        if self.res_dropout_p:
            self.dropout = nn.Dropout(self.res_dropout_p)


    def forward(self, x: Tensor) -> Tensor:
        
        # residual block with two convolutions
        conv1_outputs = [conv_layer(x) for conv_layer in self.conv1_layers]
        out = torch.cat(conv1_outputs, dim=1)
        out = self.bn1(out)
        out = self.elu1(out)

        conv2_outputs = [conv_layer(out) for conv_layer in self.conv2_layers]
        out = torch.cat(conv2_outputs, dim=1)
        out = self.bn2(out)

        # residual connection 
        res_outputs = [conv_layer(x) for conv_layer in self.stream_conv_layers]
        res = torch.cat(res_outputs, dim=1)
        res = self.stream_bn(res)

        out = out + res 

        if self.pool_size>0:
            out = self.maxpool(out)

        if self.res_dropout_p:
            out = self.dropout(out)
            
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
class Epoch_Classifier_Head(BaseModel):
    def __init__(self, in_dim, dim, out_dim, dropout_p=0.0):
        super(Epoch_Classifier_Head, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(dim, out_dim, bias=False)
        )
    def forward(self, x):
        return self.layers(x)
    
class Epoch_Classifier_Head_L(BaseModel):
    def __init__(self, in_dim, dim, out_dim, dropout_p=0.0):
        super(Epoch_Classifier_Head_L, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(dim, out_dim, bias=False)
        )

    def forward(self, x):
        return self.layers(x)

class Text_Projector(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Text_Projector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)
        )

    def forward(self, x):
        return self.layers(x)
