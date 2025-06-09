from LIM.data.structures.pair import Pair
from LIM.models.PREDATOR import Encoder, BottleNeck, Decoder
from debug.decorators import identify_method
from typing import Tuple
from LIM.models.modelI import Model
import torch


class PREDATOR(Model):
    def __init__(self) -> None:
        super(PREDATOR, self).__init__()
        self.encoder = Encoder()
        self.bottleneck = BottleNeck()
        self.decoder = Decoder()

    def __repr__(self) -> str:
        return f"Predator({self.encoder}, {self.bottleneck}, {self.decoder})"

    @identify_method
    def forward(self, pair: Pair) -> Tuple[Pair, torch.Tensor, torch.Tensor]:
        source, target = pair.source, pair.target
        (source, source_skip), (target, target_skip) = self.encoder(source), self.encoder(target)
        source, target = self.bottleneck(source, target)
        (source, source_overlap, source_saliency), (target, target_overlap, target_saliency) = (
            self.decoder(source, source_skip),
            self.decoder(target, target_skip),
        )
        pair.source, pair.target = source, target
        cat_dim = int(torch.argmax(torch.tensor(source.points.shape)).item())
        return (
            pair,
            torch.cat((source_overlap, target_overlap), dim=cat_dim),  # overlap_score
            torch.cat((source_saliency, target_saliency), dim=cat_dim),  # saliency_score
        )

    # @identify_method
    # def forward(self, pair: Pair) -> Pair:
    #     pair.join()
    #     pair, skip_connections = self.encoder(pair)
    #     pair = self.bottleneck(pair)
    #     (pair, overlap_score, saliency_score) = self.decoder(pair, skip_connections)
    #     pair.split()
    #     return pair, overlap_score, saliency_score
