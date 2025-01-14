import torch
from LIM.models.PREDATOR import Conv1D, KPConvNeighbors, KPConvPools
from LIM.models.PREDATOR.leakyrelu import LeakyReluAdapter
from LIM.models.PREDATOR.maxpool import MaxPoolNeighbors
from LIM.data.structures import Cloud

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

    def __init__(self, in_dim: int, out_dim: int, neighbor_radius: float, sampleDL: float) -> None:
        super(ResBlock_A, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.leaky_relu = LeakyReluAdapter(negative_slope=0.1)

        # These //2 and 4* differ from the values in the paper
        self.layers = torch.nn.Sequential(
            Conv1D(in_dim=in_dim, out_dim=out_dim // 4, with_batch_norm=True, with_leaky_relu=True),
            KPConvPools(
                in_dim=out_dim // 4,
                out_dim=out_dim // 4,
                neighbor_radius=neighbor_radius,
                sampleDL=sampleDL,
                KP_radius=0.06,
                KP_extent=0.05,
                n_kernel_points=15,
            ),
            self.leaky_relu,
            Conv1D(in_dim=out_dim // 4, out_dim=out_dim, with_batch_norm=True, with_leaky_relu=True),
        )

        self.shortcut = torch.nn.Sequential(
            MaxPoolNeighbors(),
            Conv1D(in_dim=in_dim, out_dim=out_dim, with_batch_norm=True, with_leaky_relu=True),
        )

    def __repr__(self) -> str:
        return f"ResBlock_A(in_dim: {self.in_dim}, out_dim: {self.out_dim})"

    def forward(self, cloud: Cloud) -> Cloud:
        print("------------------RESBLOCK_A------------------")
        print(f"resblock_A forward: ({cloud.shape}, {cloud.features.shape})")

        temp_cloud = Cloud.from_tensor(cloud.tensor.clone())
        temp_cloud.features = cloud.features.clone()

        cloud.features = self.layers(cloud).features + self.shortcut(temp_cloud).features
        cloud = self.leaky_relu(cloud)
        print("----------------------------------------------")
        return cloud


class ResBlock_B(torch.nn.Module):
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int, out_dim: int, neighbor_radius: float) -> None:
        super(ResBlock_B, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.neighbor_radius = neighbor_radius
        self.leaky_relu = LeakyReluAdapter(negative_slope=0.1)
        self.main = torch.nn.Sequential(
            Conv1D(in_dim=in_dim, out_dim=out_dim // 4, with_batch_norm=True, with_leaky_relu=True),
            KPConvNeighbors(
                in_dim=out_dim // 4,
                out_dim=out_dim // 4,
                neighbor_radius=neighbor_radius,
                KP_radius=0.06,
                KP_extent=0.05,
                n_kernel_points=15,
            ),
            self.leaky_relu,
            Conv1D(in_dim=out_dim // 4, out_dim=out_dim, with_batch_norm=True, with_leaky_relu=True),
        )
        self.shortcut = torch.nn.Sequential(
            Conv1D(in_dim=in_dim, out_dim=out_dim, with_batch_norm=True, with_leaky_relu=True),
        )

    def __repr__(self) -> str:
        return f"ResBlock_B(in_dim: {self.in_dim}, out_dim: {self.out_dim})"

    def forward(self, cloud: Cloud) -> Cloud:
        # print("------------------RESBLOCK_B------------------")
        # print(f"resblock_B forward: ({cloud.shape}, {cloud.features.shape})")

        temp_cloud = Cloud.from_tensor(cloud.tensor.clone())
        temp_cloud.features = cloud.features.clone()

        cloud.features = self.main(cloud).features + self.shortcut(temp_cloud).features
        cloud = self.leaky_relu(cloud)
        # print("----------------------------------------------")
        return cloud
