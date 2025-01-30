import torch
from LIM.data.structures.pair import Pair
from LIM.models.PREDATOR import Encoder, BottleNeck, Decoder
from debug.decorators import identify_method
from debug.context import inspect_cloud


class Predator(torch.nn.Module):
    def __init__(self) -> None:
        super(Predator, self).__init__()
        self.encoder = Encoder()
        self.bottleneck = BottleNeck()
        self.decoder = Decoder()

    @identify_method
    def forward(self, pair: Pair) -> Pair:
        source, target = pair.source, pair.target
        (source, source_skip), (target, target_skip) = self.encoder(source), self.encoder(target)
        source, target = self.bottleneck(source, target)
        source, target = self.decoder(source, source_skip), self.decoder(target, target_skip)

        src, tgt = source.superpoints, target.superpoints
        while (src is not None) and (tgt is not None):
            src.features, tgt.features = src.features.detach(), tgt.features.detach()
            src, tgt = src.superpoints, tgt.superpoints

        pair.source, pair.target = source, target
        return pair
