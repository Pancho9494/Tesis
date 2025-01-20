import torch
from LIM.data.structures.cloud import Cloud
from debug.decorators import identify_method

class LeakyRelU(torch.nn.Module):
    negative_slope: float

    def __init__(self, negative_slope: float) -> None:
        super(LeakyRelU, self).__init__()
        self.negative_slope = negative_slope
        self._relu = torch.nn.LeakyReLU(negative_slope=self.negative_slope)

    def __repr__(self) -> str:
        return f"LeakyRelU(negative_slope: {self.negative_slope})"
    
    @identify_method
    def forward(self, cloud: Cloud) -> Cloud:
        cloud.features = self._relu(cloud.features)
        return cloud
