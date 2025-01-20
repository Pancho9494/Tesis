import torch
from LIM.data.structures import Cloud
from debug.decorators import identify_method

class Conv2D(torch.nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    bias: bool
    _conv2d: torch.nn.Module
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool) -> None:
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self._conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
    
    def __repr__(self) -> str:
        out = f"Conv2D(in_channels: {self.in_channels}, out_channels: {self.out_channels}, "
        out += f"kernel_size: {self.kernel_size}), bias: {self.bias})"
        return out
    
    @identify_method
    def forward(self, cloud: Cloud) -> Cloud:
        cloud.features = self._conv2d(cloud.features)
        return cloud