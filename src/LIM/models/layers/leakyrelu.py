import torch
from LIM.data.structures.pcloud import PCloud


class LeakyReLU(torch.nn.Module):
    negative_slope: float

    def __init__(self, negative_slope: float) -> None:
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self._relu = torch.nn.LeakyReLU(negative_slope=self.negative_slope)

    def __repr__(self) -> str:
        return f"LeakyRelU(negative_slope: {self.negative_slope})"

    def forward(self, cloud: PCloud) -> PCloud:
        cloud.features = self._relu(cloud.features)
        return cloud
