import torch
from LIM.data.structures.cloud import Cloud


class LeakyReluAdapter(torch.nn.Module):
    negative_slope: float

    def __init__(self, negative_slope: float) -> None:
        super(LeakyReluAdapter, self).__init__()
        self.negative_slope = negative_slope
        self._relu = torch.nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, cloud: Cloud) -> Cloud:
        # print(f"leakyreluadapter forward: ({cloud.shape}, {cloud.features.shape})")
        cloud.features = self._relu(cloud.features)
        return cloud
