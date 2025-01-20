import torch
from typing import Tuple, List
from LIM.data.structures.cloud import Cloud
from LIM.models.PREDATOR.blocks import KPConvNeighbors, ResBlock_A, ResBlock_B, Conv1D
from LIM.models.PREDATOR.blocks.leakyrelu import LeakyRelU

from debug.decorators import identify_method

class Encoder(torch.nn.Module):
    skip_connections: List[torch.Tensor]
    
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.skip_connections = []
        BLOCK_1_RADIUS, BLOCK_1_DL = 0.0625, 0.05
        self.block1 = torch.nn.ModuleList(
            [
                KPConvNeighbors(in_dim=1, out_dim=64, neighbor_radius=BLOCK_1_RADIUS, sampleDL=BLOCK_1_DL),
                LeakyRelU(negative_slope=0.1),
                ResBlock_B(in_dim=64, out_dim=128, neighbor_radius=BLOCK_1_RADIUS, sampleDL=BLOCK_1_DL),
                ResBlock_A(in_dim=128, out_dim=128, neighbor_radius=BLOCK_1_RADIUS, sampleDL=BLOCK_1_DL),
            ]
        )

        BLOCK_2_RADIUS, BLOCK_2_DL = 0.125, 0.1
        self.block2 = torch.nn.ModuleList(
            [
                ResBlock_B(in_dim=128, out_dim=256, neighbor_radius=BLOCK_2_RADIUS, sampleDL=BLOCK_2_DL),
                ResBlock_B(in_dim=256, out_dim=256, neighbor_radius=BLOCK_2_RADIUS, sampleDL=BLOCK_2_DL),
                ResBlock_A(in_dim=256, out_dim=256, neighbor_radius=BLOCK_2_RADIUS, sampleDL=BLOCK_2_DL),
            ]
        )

        BLOCK_3_RADIUS, BLOCK_3_DL = 0.25, 0.2
        self.block3 = torch.nn.ModuleList(
            [
                ResBlock_B(in_dim=256, out_dim=512, neighbor_radius=BLOCK_3_RADIUS, sampleDL=BLOCK_3_DL),
                ResBlock_B(in_dim=512, out_dim=512, neighbor_radius=BLOCK_3_RADIUS, sampleDL=BLOCK_3_DL),
                ResBlock_A(in_dim=512, out_dim=512, neighbor_radius=BLOCK_3_RADIUS, sampleDL=BLOCK_3_DL),
            ]
        )

        BLOCK_4_RADIUS, BLOCK_4_DL = 0.5, None
        self.block4 = torch.nn.ModuleList(
            [
                ResBlock_B(in_dim=512, out_dim=1024, neighbor_radius=BLOCK_4_RADIUS, sampleDL=BLOCK_4_DL),
                ResBlock_B(in_dim=1024, out_dim=1024, neighbor_radius=BLOCK_4_RADIUS, sampleDL=BLOCK_4_DL),
                Conv1D(in_dim=1024, out_dim=256, with_batch_norm=True, with_leaky_relu=True),
            ]
        )
        
    def __repr__(self) -> str:
        return f"Encoder()"

    @identify_method
    def forward(self, cloud: Cloud) -> Tuple[Cloud, List[torch.Tensor]]:
        skip_connections: List[torch.Tensor] = []
        for idx, block in enumerate([self.block1, self.block2, self.block3]):
            for layer in block:
                cloud = layer(cloud)
            skip_connections.append(cloud.features)

        for layer in self.block4:
            cloud = layer(cloud)
        return cloud, skip_connections
