import torch
from LIM.data.structures import PCloud
from debug.decorators import identify_method


class ReLU(torch.nn.Module):
    _ReLU: torch.nn.Module

    def __init__(self) -> None:
        super(ReLU, self).__init__()
        self._ReLU = torch.nn.ReLU()

    def __repr__(self) -> str:
        return "ReLU()"

    @identify_method
    def forward(self, cloud: PCloud) -> PCloud:
        cloud.features = self._ReLU(cloud.features)
        return cloud
