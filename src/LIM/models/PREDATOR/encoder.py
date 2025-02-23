import torch
from typing import Tuple, List

from LIM.data.structures import Pair
from LIM.models.PREDATOR.blocks import KPConvNeighbors, ResBlock_A, ResBlock_B, Conv1DAdapter, BatchNorm
from LIM.models.PREDATOR.blocks.leakyrelu import LeakyReLU
from debug.decorators import identify_method


class Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.neighbor_radius = [(2**i) * 0.0625 for i in range(4)]
        self.sample_dl = [(2**i) * 0.05 for i in range(3)] + [None]

        print(self.neighbor_radius)
        print(self.sample_dl)

        self.block1 = torch.nn.Sequential(
            # 0 simple
            KPConvNeighbors(in_dim=1, out_dim=64, radius=self.neighbor_radius[0]),
            BatchNorm(in_dim=64, momentum=0.02),
            LeakyReLU(negative_slope=0.1),
            # 1 resnetb
            ResBlock_B(in_dim=64, out_dim=128, radius=self.neighbor_radius[0]),
            # 2 resnetb_strided
            ResBlock_A(in_dim=128, out_dim=128, radius=self.neighbor_radius[0]),
        )
        self.block2 = torch.nn.Sequential(
            # 3 resnetb
            ResBlock_B(in_dim=128, out_dim=256, radius=self.neighbor_radius[1]),
            # 4 resnetb
            ResBlock_B(in_dim=256, out_dim=256, radius=self.neighbor_radius[1]),
            # 5 resnetb_strided
            ResBlock_A(in_dim=256, out_dim=256, radius=self.neighbor_radius[1]),
        )
        self.block3 = torch.nn.Sequential(
            # 6 resnetb
            ResBlock_B(in_dim=256, out_dim=512, radius=self.neighbor_radius[2]),
            # 7 resnetb
            ResBlock_B(in_dim=512, out_dim=512, radius=self.neighbor_radius[2]),
            # 8 resnetb_strided
            ResBlock_A(in_dim=512, out_dim=512, radius=self.neighbor_radius[2]),
        )
        self.block4 = torch.nn.Sequential(
            # 9 resnetb
            ResBlock_B(in_dim=512, out_dim=1024, radius=self.neighbor_radius[3]),
            # 10 resnetb
            ResBlock_B(in_dim=1024, out_dim=1024, radius=self.neighbor_radius[3]),
            # 11 bottle
            Conv1DAdapter(in_channels=1024, out_channels=256, kernel_size=1, bias=True, debug_mode=True),
        )

    def __repr__(self) -> str:
        return "Encoder()"

    @multimethod
    def forward(self, *args, **kwargs) -> Any: ...

    @identify_method
    @multimethod
    def forward(self, cloud: PCloud) -> Tuple[PCloud, List[torch.Tensor]]:
        skip_connections: List[torch.Tensor] = []

        current: PCloud = cloud
        for idx, block in enumerate([self.block1, self.block2, self.block3]):
            current.compute_neighbors(self.neighbor_radius[idx], self.sample_dl[idx])
            super_cloud = block(copy.copy(current))
            skip_connections.append(block[-1].skip_connection)
            current._super.features = super_cloud.features.clone()
            current = current._super

        current.compute_neighbors(self.neighbor_radius[-1], self.sample_dl[-1])
        super_cloud = self.block4(copy.copy(current))
        current._super.features = super_cloud.features.clone()
        return current, skip_connections

    @identify_method
    @multimethod
    def forward(self, pair: Pair) -> Tuple[Pair, List[torch.Tensor]]:
        skip_connections: List[torch.Tensor] = []

        current: Pair = pair

        # TODO: pre-compute the neighbors by injecting Encoder into Pair.compute_neighbors
        for idx, block in enumerate([self.block1, self.block2, self.block3]):
            current.compute_neighbors(self.neighbor_radius[idx], self.sample_dl[idx])
            current.mix = block(current.mix)
            skip_connections.append(block[-1].skip_connection)
            current.mix._super.features = current.mix.features
            current.mix = current.mix._super
            current.source = current.source._super if current.source._super is not None else current.source
            current.target = current.target._super if current.target._super is not None else current.target

        current.compute_neighbors(self.neighbor_radius[-1], self.sample_dl[-1])
        current.mix = self.block4(current.mix)
        del current.mix._super
        current.mix.features = current.mix.features.transpose(0, 1).unsqueeze(0)
        return current, skip_connections
