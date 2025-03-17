from LIM.data.structures.pair import Pair
from LIM.models.PREDATOR import Encoder, BottleNeck, Decoder
from debug.decorators import identify_method
from multimethod import multimethod
from typing import Any
from LIM.models.modelI import Model


class PREDATOR(Model):
    def __init__(self) -> None:
        super(PREDATOR, self).__init__()
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
        pair = self.encoder(pair)
        pair = self.bottleneck(pair)
        pair = self.decoder(pair, self.encoder.skip_connections)
        return pair
