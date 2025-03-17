import torch
from typing import Optional
from LIM.models.blocks import KNNGraph, MaxPool, Conv2D, InstanceNorm2D
from LIM.models.blocks.leakyrelu import LeakyReLU
from LIM.data.structures.pcloud import PCloud
from debug.decorators import identify_method


class EdgeConv(torch.nn.Module):
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int, out_dim: int, maxPool: bool, knn: Optional[int] = None) -> None:
        super(EdgeConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = torch.nn.Sequential(
            KNNGraph(knn=knn) if knn is not None else torch.nn.Identity(),
            Conv2D(in_channels=in_dim, out_channels=out_dim, kernel_size=1, bias=False),
            InstanceNorm2D(num_features=out_dim),
            LeakyReLU(negative_slope=0.2),
            MaxPool() if maxPool else torch.nn.Identity(),
        )

    def __repr__(self) -> str:
        return f"EdgeConv(in_dim: {self.in_dim}, out_dim: {self.out_dim})"

    @identify_method
    def forward(self, cloud: PCloud) -> PCloud:
        cloud = self.layers(cloud)
        return cloud
