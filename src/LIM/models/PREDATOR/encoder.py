import torch
from typing import List, Any, Tuple
from multimethod import multimethod
from LIM.data.structures import PCloud, Pair
from LIM.models.layers import KPConvNeighbors, ResBlock_A, ResBlock_B, Conv1DAdapter, BatchNorm
from LIM.models.layers.leakyrelu import LeakyReLU
from config.config import settings


class Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()

        N_LAYERS = settings.MODEL.ENCODER.N_HIDDEN_LAYERS
        LATENT_DIM = settings.MODEL.LATENT_DIM
        self.neighbor_radius = [(2**i) * 0.0625 for i in range(2 + N_LAYERS)]
        self.sample_dl = [(2**i) * 0.05 for i in range(1 + N_LAYERS)] + [None]
        self.enter = torch.nn.Sequential(
            KPConvNeighbors(in_dim=2 ** (0), out_dim=2 ** (6), radius=self.neighbor_radius[0]),
            BatchNorm(in_dim=2 ** (6), momentum=0.02),
            LeakyReLU(negative_slope=0.1),
            ResBlock_B(in_dim=2 ** (6), out_dim=2 ** (7), radius=self.neighbor_radius[0]),
            ResBlock_A(in_dim=2 ** (7), out_dim=2 ** (7), radius=self.neighbor_radius[0]),
        )

        self.inner_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    ResBlock_B(in_dim=2 ** (idx + 0), out_dim=2 ** (idx + 1), radius=self.neighbor_radius[idx - 6]),
                    ResBlock_B(in_dim=2 ** (idx + 1), out_dim=2 ** (idx + 1), radius=self.neighbor_radius[idx - 6]),
                    ResBlock_A(in_dim=2 ** (idx + 1), out_dim=2 ** (idx + 1), radius=self.neighbor_radius[idx - 6]),
                )
                for idx in range(7, 7 + N_LAYERS)
            ]
        )

        self.exit = torch.nn.Sequential(
            ResBlock_B(in_dim=2 ** (7 + N_LAYERS), out_dim=2 ** (8 + N_LAYERS), radius=self.neighbor_radius[-1]),
            ResBlock_B(in_dim=2 ** (8 + N_LAYERS), out_dim=2 ** (8 + N_LAYERS), radius=self.neighbor_radius[-1]),
            Conv1DAdapter(
                in_channels=2 ** (8 + N_LAYERS), out_channels=LATENT_DIM, kernel_size=1, bias=True, debug_mode=True
            ),
        )

    def __repr__(self) -> str:
        return "Encoder()"

    @multimethod
    def forward(self, *args, **kwargs) -> Any: ...

    @multimethod
    def forward(self, cloud: PCloud) -> Tuple[PCloud, List[torch.Tensor]]:
        skip_connections: List[torch.Tensor] = []
        current: PCloud = cloud
        for idx, block in enumerate([self.enter, *self.inner_layers]):
            current.compute_neighbors(self.neighbor_radius[idx], self.sample_dl[idx])
            current = block(current)
            skip_connections.append(block[-1].skip_connection)
            current._super.features = current.features

            current = current._super

        current.compute_neighbors(self.neighbor_radius[-1], self.sample_dl[-1])
        current = self.exit(current)
        del current._super
        current.features = current.features.transpose(0, 1).unsqueeze(0)
        return current, skip_connections

    @multimethod
    def forward(self, pair: Pair) -> Tuple[Pair, List[torch.Tensor]]:
        skip_connections: List[torch.Tensor] = []
        current: Pair = pair
        for idx, block in enumerate([self.enter, *self.inner_layers]):
            current.compute_neighbors(self.neighbor_radius[idx], self.sample_dl[idx])
            current.mix = block(current.mix)
            skip_connections.append(block[-1].skip_connection)
            current.mix._super.features = current.mix.features

            current.mix = current.mix._super
            current.source = current.source._super if current.source._super is not None else current.source
            current.target = current.target._super if current.target._super is not None else current.target

        current.compute_neighbors(self.neighbor_radius[-1], self.sample_dl[-1])
        current.mix = self.exit(current.mix)
        del current.mix._super
        current.mix.features = current.mix.features.transpose(0, 1).unsqueeze(0)
        return current, skip_connections
