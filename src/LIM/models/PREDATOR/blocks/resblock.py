import torch
from LIM.models.PREDATOR.blocks import BatchNorm, Conv1D, KPConvNeighbors, KPConvPools
from LIM.models.PREDATOR.blocks.leakyrelu import LeakyReLU
from LIM.models.PREDATOR.blocks.maxpool import MaxPoolNeighbors
from LIM.data.structures import PCloud
from copy import copy
from debug.decorators import identify_method

"""
These two were supposed to be different according to the paper

But in the author's implementation the only difference is that ResBlock_A 'resnetb_strided' has a maxpool in the
shortcut and that kpconv uses the pools instead of the neighbors

Also the dimensions are weird, according to the paper the intermediate layers should be of size N // 2 and the 
output should be of 2 * N
"""


class ResBlock_A(torch.nn.Module):
    in_dim: int
    out_dim: int
    radius: float
    skip_connection: torch.Tensor

    def __init__(self, in_dim: int, out_dim: int, radius: float) -> None:
        super(ResBlock_A, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.radius = radius
        self._leaky_relu = LeakyReLU(negative_slope=0.1)

        # These //2 and 4* differ from the values in the paper
        self.layers = torch.nn.Sequential(
            Conv1D(in_dim=in_dim, out_dim=out_dim // 4, with_batch_norm=True, with_leaky_relu=True),
            KPConvPools(
                in_dim=out_dim // 4,
                out_dim=out_dim // 4,
                radius=radius,
                n_kernel_points=15,
            ),
            BatchNorm(in_dim=out_dim // 4, momentum=0.02),
            self._leaky_relu,
            Conv1D(in_dim=out_dim // 4, out_dim=out_dim, with_batch_norm=False, with_leaky_relu=False),
        )

        self.shortcut = torch.nn.Sequential(
            MaxPoolNeighbors(),
            Conv1D(in_dim=in_dim, out_dim=out_dim, with_batch_norm=True, with_leaky_relu=False)
            if in_dim != out_dim
            else torch.nn.Identity(),
        )

    def __repr__(self) -> str:
        return f"ResBlock_A(in_dim: {self.in_dim}, out_dim: {self.out_dim}, radius={self.radius})"

    @identify_method
    def forward(self, cloud: PCloud) -> PCloud:
        self.skip_connection = cloud.features.clone()
        cloud.features = self.shortcut(copy(cloud)).features + self.layers(cloud).features
        return self._leaky_relu(cloud)


class ResBlock_B(torch.nn.Module):
    in_dim: int
    out_dim: int
    radius: float

    def __init__(self, in_dim: int, out_dim: int, radius: float) -> None:
        super(ResBlock_B, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.radius = radius
        self.leaky_relu = LeakyReLU(negative_slope=0.1)
        self.main = torch.nn.Sequential(
            Conv1D(in_dim=in_dim, out_dim=out_dim // 4, with_batch_norm=True, with_leaky_relu=True),
            KPConvNeighbors(
                in_dim=out_dim // 4,
                out_dim=out_dim // 4,
                radius=radius,
                n_kernel_points=15,
            ),
            BatchNorm(in_dim=out_dim // 4, momentum=0.02),
            self.leaky_relu,
            Conv1D(in_dim=out_dim // 4, out_dim=out_dim, with_batch_norm=True, with_leaky_relu=False),
        )
        self.shortcut = torch.nn.Sequential(
            Conv1D(in_dim=in_dim, out_dim=out_dim, with_batch_norm=True, with_leaky_relu=False)
            if in_dim != out_dim
            else torch.nn.Identity(),
        )

    def __repr__(self) -> str:
        return f"ResBlock_B(in_dim: {self.in_dim}, out_dim: {self.out_dim}, radius={self.radius})"

    @identify_method
    def forward(self, cloud: PCloud) -> PCloud:
        cloud.features = self.shortcut(copy(cloud)).features + self.main(cloud).features
        return self.leaky_relu(cloud)
