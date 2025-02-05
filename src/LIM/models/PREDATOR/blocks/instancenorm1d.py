import torch
from LIM.data.structures import Cloud
from debug.decorators import identify_method

class InstanceNorm1D(torch.nn.Module):
    num_features: int
    _instancenorm1D: torch.nn.Module
    
    def __init__(self, num_features: int) -> None:
        super(InstanceNorm1D, self).__init__()
        self.num_features = num_features
        self._instancenorm1D = torch.nn.InstanceNorm1d(num_features)
    
    def __repr__(self) -> str:
        return f"InstanceNorm1D(num_features: {self.num_features})"
    
    def forward(self, cloud: Cloud) -> Cloud:
        cloud.features = self._instancenorm1D(cloud.features)
        return cloud