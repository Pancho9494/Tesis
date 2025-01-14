import torch
from typing import Optional
from LIM.models.PREDATOR import KNNGraph, MaxPool
from LIM.data.structures.cloud import Cloud


class EdgeConv(torch.nn.Module):
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int, out_dim: int, maxPool: bool, knn: Optional[int] = None) -> None:
        super(EdgeConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = torch.nn.Sequential(
            KNNGraph(knn=knn) if knn is not None else torch.nn.Identity(),
            torch.nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, bias=False),
            torch.nn.InstanceNorm2d(num_features=out_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            MaxPool() if maxPool else torch.nn.Identity(),
        )

    def forward(self, cloud: Cloud) -> Cloud:
        cloud = self.layers(cloud)
        return cloud
