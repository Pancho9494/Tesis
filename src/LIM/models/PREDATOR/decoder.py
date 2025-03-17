import torch
from typing import List
from LIM.data.structures import Pair
from LIM.models.PREDATOR.blocks import Conv1D
from LIM.models.PREDATOR.blocks.nearestupsample import NearestUpsample
import config


class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.nearest_upsample = NearestUpsample()
        enter_dim = config.settings.MODEL.LATENT_DIM + 2
        self.inner_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    Conv1D(
                        in_dim=(enter_dim := enter_dim + 2 ** (idx)),
                        out_dim=(enter_dim := enter_dim // 6),
                        with_batch_norm=True,
                        with_leaky_relu=True,
                    )
                )
                for idx in range(7 + config.settings.MODEL.ENCODER.N_HIDDEN_LAYERS, 7, -1)
            ]
        )

        self.exit = torch.nn.Sequential(
            Conv1D(
                in_dim=(enter_dim := enter_dim + 2 ** (7)),
                out_dim=enter_dim // 6,
                with_batch_norm=False,
                with_leaky_relu=False,
            )
        )

    def __repr__(self) -> str:
        return "Decoder()"

    def forward(self, pair: Pair, skip_connections: List[torch.Tensor]) -> Pair:
        for block, skip in zip([*self.inner_layers, self.exit], reversed(skip_connections)):
            pair.mix = self.nearest_upsample(pair.mix)
            pair.mix.features = torch.cat([pair.mix.features, skip], dim=1)
            pair.mix = block(pair.mix)
            pair.mix._sub.features = pair.mix.features
            pair.mix = pair.mix._sub
            pair.source = pair.source._sub
            pair.target = pair.target._sub

        pair.set_overlaps_saliencies(pair.mix.features, self.exit[0].out_dim - 2)
        return pair
