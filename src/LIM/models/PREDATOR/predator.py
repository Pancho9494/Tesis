from typing import Tuple

import torch

import LIM.log as log
from config.config import settings
from LIM.data.structures.pair import Pair
from LIM.models.IAE import IAE
from LIM.models.modelI import Model
from LIM.models.PREDATOR import BottleNeck, Decoder, Encoder


class PREDATOR(Model):
    def __init__(self) -> None:
        log.info("Calling PREDATOR.__init__")
        super(PREDATOR, self).__init__()
        self.encoder = Encoder()
        self.bottleneck = BottleNeck()
        self.decoder = Decoder()

        self._load_pre_training()

    def __repr__(self) -> str:
        return f"Predator({self.encoder}, {self.bottleneck}, {self.decoder})"

    def _load_pre_training(self) -> None:
        if not settings.MODEL.ENCODER.PRE_TRAINED:
            return
        run_path = settings.TRAINER.BACKUP_DIR / "IAE" / settings.MODEL.ENCODER.PRE_TRAIN_DATE
        log.info(f"Loading pre_trained weights from: {run_path}")
        pre_training_IAE = IAE(model=self)
        pre_training_IAE.load(run=run_path, suffix="best")

        self.encoder.load_state_dict(pre_training_IAE.encoder.state_dict())
        log.info("Successfuly loaded encoder weights")

        if settings.MODEL.ENCODER.FREEZE:
            log.info("Freezing encoder weights")
            self.encoder.requires_grad_(False)

        del pre_training_IAE
        return

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
