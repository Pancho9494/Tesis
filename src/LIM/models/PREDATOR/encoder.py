import torch
from typing import Tuple, List
from LIM.data.structures.cloud import Cloud
from LIM.models.PREDATOR import KPConvNeighbors, ResBlock_A, ResBlock_B, Conv1D
from LIM.models.PREDATOR.leakyrelu import LeakyReluAdapter


class Encoder(torch.nn.Module):
    skip_connections: List[torch.Tensor]

    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.skip_connections = []
        BLOCK_1_RADIUS, BLOCK_1_DL = 0.0625, 0.05
        self.block1 = torch.nn.ModuleList(
            [
                KPConvNeighbors(in_dim=1, out_dim=64, neighbor_radius=BLOCK_1_RADIUS),
                LeakyReluAdapter(negative_slope=0.1),
                ResBlock_B(in_dim=64, out_dim=128, neighbor_radius=BLOCK_1_RADIUS),
                ResBlock_A(in_dim=128, out_dim=128, neighbor_radius=BLOCK_1_RADIUS, sampleDL=BLOCK_1_DL),
            ]
        )

        BLOCK_2_RADIUS, BLOCK_2_DL = 0.125, 0.1
        self.block2 = torch.nn.ModuleList(
            [
                ResBlock_B(in_dim=128, out_dim=256, neighbor_radius=BLOCK_2_RADIUS),
                ResBlock_B(in_dim=256, out_dim=256, neighbor_radius=BLOCK_2_RADIUS),
                ResBlock_A(in_dim=256, out_dim=256, neighbor_radius=BLOCK_2_RADIUS, sampleDL=BLOCK_2_DL),
            ]
        )

        BLOCK_3_RADIUS, BLOCK_3_DL = 0.25, 0.2
        self.block3 = torch.nn.ModuleList(
            [
                ResBlock_B(in_dim=256, out_dim=512, neighbor_radius=BLOCK_3_RADIUS),
                ResBlock_B(in_dim=512, out_dim=512, neighbor_radius=BLOCK_3_RADIUS),
                ResBlock_A(in_dim=512, out_dim=512, neighbor_radius=BLOCK_3_RADIUS, sampleDL=BLOCK_3_DL),
            ]
        )

        BLOCK_4_RADIUS, BLOCK_4_DL = 0.5, None
        self.block4 = torch.nn.ModuleList(
            [
                ResBlock_B(in_dim=512, out_dim=1024, neighbor_radius=BLOCK_4_RADIUS),
                ResBlock_B(in_dim=1024, out_dim=1024, neighbor_radius=BLOCK_4_RADIUS),
                Conv1D(in_dim=1024, out_dim=256, with_batch_norm=True, with_leaky_relu=True),
            ]
        )

    def forward(self, cloud: Cloud) -> Tuple[Cloud, List[torch.Tensor]]:
        print(f"encoder forward ({cloud.shape}, {cloud.features.shape})")
        skip_connections: List[torch.Tensor] = []

        print(cloud.shape, cloud.features.shape)
        for idx, block in enumerate([self.block1, self.block2, self.block3]):
            print(f"===== BLOCK {idx + 1} ========")
            for layer in block:
                cloud = layer(cloud)
            print(cloud.shape, cloud.features.shape)
            skip_connections.append(cloud.features)

        print("===== BLOCK 4 ========")
        for layer in self.block4:
            cloud = layer(cloud)
        print(cloud.shape, cloud.features.shape)
        print()
        return cloud, skip_connections
