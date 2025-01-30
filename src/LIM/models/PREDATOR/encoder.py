import torch
from typing import Tuple, List
from LIM.data.structures.cloud import Cloud
from LIM.models.PREDATOR.blocks import KPConvNeighbors, ResBlock_A, ResBlock_B, Conv1D
from LIM.models.PREDATOR.blocks.leakyrelu import LeakyRelU

from debug.decorators import identify_method
from debug.context import inspect_cloud


class Encoder(torch.nn.Module):
    skip_connections: List[torch.Tensor]

    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.skip_connections = []
        self.block1 = torch.nn.ModuleList(
            [
                KPConvNeighbors(in_dim=1, out_dim=64),
                LeakyRelU(negative_slope=0.1),
                ResBlock_B(in_dim=64, out_dim=128),
                ResBlock_A(in_dim=128, out_dim=128),
            ]
        )
        self.block2 = torch.nn.ModuleList(
            [
                ResBlock_B(in_dim=128, out_dim=256),
                ResBlock_B(in_dim=256, out_dim=256),
                ResBlock_A(in_dim=256, out_dim=256),
            ]
        )
        self.block3 = torch.nn.ModuleList(
            [
                ResBlock_B(in_dim=256, out_dim=512),
                ResBlock_B(in_dim=512, out_dim=512),
                ResBlock_A(in_dim=512, out_dim=512),
            ]
        )
        self.block4 = torch.nn.ModuleList(
            [
                ResBlock_B(in_dim=512, out_dim=1024),
                ResBlock_B(in_dim=1024, out_dim=1024),
                Conv1D(in_dim=1024, out_dim=256, with_batch_norm=True, with_leaky_relu=True),
            ]
        )

    def __repr__(self) -> str:
        return "Encoder()"

    @identify_method
    def forward(self, cloud: Cloud) -> Tuple[Cloud, List[torch.Tensor]]:
        skip_connections: List[torch.Tensor] = []
        cloud.tensor, cloud.features = cloud.tensor, cloud.features

        NEIGHBOR_RADIUS = [(2**i) * 0.0625 for i in range(4)]
        SAMPLE_DL = [(2**i) * 0.05 for i in range(3)] + [None]

        current: Cloud = cloud
        # inspect_cloud(current)
        for idx, block in enumerate([self.block1, self.block2, self.block3]):
            current.compute_neighbors(NEIGHBOR_RADIUS[idx], SAMPLE_DL[idx])
            for layer in block:
                current = layer(current)
            # inspect_cloud(current)
            skip_connections.append(layer.skip_connection)  # TODO: I'm not sure about this solution

            current.superpoints.features = current.features.clone()
            current = current.superpoints

        current.compute_neighbors(NEIGHBOR_RADIUS[-1], SAMPLE_DL[-1])
        for layer in self.block4:
            current = layer(current)
        # inspect_cloud(current)

        return current, skip_connections
