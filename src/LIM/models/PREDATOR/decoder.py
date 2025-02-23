import torch
from typing import List
from LIM.data.structures import Pair
from LIM.models.PREDATOR.blocks import Conv1D
from LIM.models.PREDATOR.blocks.nearestupsample import NearestUpsample


class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.nearest_upsample = NearestUpsample()
        self.block1 = torch.nn.Sequential(
            Conv1D(in_dim=770, out_dim=129, with_batch_norm=True, with_leaky_relu=True),
        )
        self.block2 = torch.nn.Sequential(
            Conv1D(in_dim=385, out_dim=64, with_batch_norm=True, with_leaky_relu=True),
        )
        self.block3 = torch.nn.Sequential(
            Conv1D(in_dim=192, out_dim=34, with_batch_norm=False, with_leaky_relu=False),
        )

    def __repr__(self) -> str:
        return "Decoder()"

    def forward(self, pair: Pair, skip_connections: List[torch.Tensor]) -> Pair:
        for block, skip in zip([self.block1, self.block2, self.block3], reversed(skip_connections)):
            pair.mix = self.nearest_upsample(pair.mix)
            pair.mix.features = torch.cat([pair.mix.features, skip], dim=1)
            pair.mix = block(pair.mix)
            pair.mix._sub.features = pair.mix.features
            pair.mix = pair.mix._sub
            pair.source = pair.source._sub
            pair.target = pair.target._sub

        pair.set_overlaps_saliencies(pair.mix.features)
        return pair
