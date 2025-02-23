import torch
from LIM.data.structures.pair import Pair
from LIM.models.PREDATOR import Encoder, BottleNeck, Decoder
from debug.decorators import identify_method
from multimethod import multimethod
from typing import Any


class Predator(torch.nn.Module):
    def __init__(self) -> None:
        super(Predator, self).__init__()
        self.encoder = Encoder()
        self.bottleneck = BottleNeck()
        self.decoder = Decoder()

    def __repr__(self) -> str:
        return f"Predator({self.encoder}, {self.bottleneck}, {self.decoder})"

    @multimethod
    def forward(self, *args, **kwargs) -> Any: ...

    @identify_method
    @multimethod
    def forward(self, pair: Pair) -> Pair:
        pair.join()
        pair, skip_connections = self.encoder(pair)
        pair = self.bottleneck(pair)
        pair = self.decoder(pair, skip_connections)
        return pair
