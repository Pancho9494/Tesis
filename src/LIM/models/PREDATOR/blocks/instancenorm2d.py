import torch
from LIM.data.structures import Cloud
from debug.decorators import identify_method

class InstanceNorm2D(torch.nn.Module):
    num_features: int
    _instancenorm2D: torch.nn.Module
    
    def __init__(self, num_features: int) -> None:
        super(InstanceNorm2D, self).__init__()
        self.num_features = num_features
        self._instancenorm2D = torch.nn.InstanceNorm2d(num_features)
    
    def __repr__(self) -> str:
        return f"InstanceNorm2D(num_features: {self.num_features})"
    
    @identify_method
    def forward(self, cloud: Cloud) -> Cloud:
        cloud.features = self._instancenorm2D(cloud.features)
        return cloud