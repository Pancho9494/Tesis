import torch
from typing import Tuple
from LIM.data.structures.pair import Pair
from LIM.models.PREDATOR import Encoder, BottleNeck, Decoder


class Predator(torch.nn.Module):
    #     Wrapper for the PREDATOR model that follows the ModelI interface

    # layer_ind
    #            00   'simple',               KPConv
    #   0        01   'resnetb',              ResnetB
    #            02   'resnetb_strided',      ResnetA  Pools(dl=0.05), Neighbors(r=0.0625)

    #            03   'resnetb',              ResnetB
    #   1        04   'resnetb',              ResnetB
    #            05   'resnetb_strided',      ResnetA  Pools(dl=0.1),  Neighbors(r=0.125)

    #            06   'resnetb',              ResnetB
    #   2        07   'resnetb',              ResnetB
    #            08   'resnetb_strided',      ResnetA  Pools(dl=0.2),  Neighbors(r=0.25)

    #            09   'resnetb',              ResnetB
    #   3        10   'resnetb',              ResnetB  zeros((0, 1)),      Neighbors(r=0.5)
    #                                     Conv1D (it's called self.bottle)

    #     11   'nearest_upsample',
    #     12   'unary',
    #     13   'nearest_upsample',
    #     14   'unary',
    #     15   'nearest_upsample',
    #     16   'last_unary'
    #     """
    def __init__(self) -> None:
        super(Predator, self).__init__()
        self.encoder = Encoder()
        self.bottleneck = BottleNeck()
        self.decoder = Decoder()

    def forward(self, pair: Pair) -> Pair:
        print(
            f"predator forward ({pair.src.shape}, {pair.src.features.shape}) ({pair.target.shape}, {pair.target.features.shape})"
        )
        source, target = pair.src, pair.target

        source.tensor, target.tensor = source.tensor.reshape(-1, 3), target.tensor.reshape(-1, 3)
        source.features, target.features = source.features.reshape(-1, 1), target.features.reshape(-1, 1)

        (source, source_skip), (target, target_skip) = self.encoder(source), self.encoder(target)

        source.tensor, target.tensor = source.tensor.reshape(1, -1, 3), target.tensor.reshape(1, -1, 3)
        source.features, target.features = (
            source.features.reshape(1, source.shape[1], -1),
            target.features.reshape(1, source.shape[1], -1),
        )

        source, target = self.bottleneck(source, target)
        source, target = self.decoder(source, source_skip), self.decoder(target, target_skip)
        return pair
