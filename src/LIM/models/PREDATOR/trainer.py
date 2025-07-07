from datetime import datetime
from typing import Type
import torch

import config.config as config
import LIM.log as log
from LIM.data.sets import CloudDatasetsI
from LIM.data.structures import Pair
from LIM.metrics import CircleLoss, FeatureMatchRecall, MatchabilityLoss, MultiLoss, OverlapLoss
from LIM.models.modelI import Model
from LIM.training.trainer import BaseTrainer, handle_OOM


class PredatorTrainer(BaseTrainer):
    multi_loss: MultiLoss
    feature_match_recall: FeatureMatchRecall
    _settings: config.Settings

    def __init__(self, model: Type[Model], dataset: Type[CloudDatasetsI], mode: BaseTrainer.Mode) -> None:
        super(PredatorTrainer, self).__init__(model, dataset, mode)
        assert config.settings is not None
        self._settings = config.settings
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self._settings.TRAINER.LEARNING_RATE.VALUE,
            weight_decay=self._settings.TRAINER.LEARNING_RATE.WEIGHT_DECAY,
            momentum=self._settings.TRAINER.LEARNING_RATE.MOMENTUM,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        self.multi_loss = MultiLoss(
            losses=[
                CircleLoss(trainer_state=self.state, weight=1.0),
                OverlapLoss(trainer_state=self.state, weight=1.0),
                MatchabilityLoss(trainer_state=self.state, weight=0.0),
            ],
        )
        self.feature_match_recall = FeatureMatchRecall(trainer_state=self.state)

    def _load_model(self, model: Type[Model]) -> None:
        self.model = model()
        self.model.to(self.device)

    @handle_OOM
    def _custom_train_step(self, sample: Pair) -> bool:
        sample.correspondences
        sample, overlaps, saliencies = self.model(sample)
        sample.source.first.pcd = sample.source.first.pcd.transform(sample.GT_tf_matrix)
        self.multi_loss.losses[1].current_overlap_score = overlaps  # TODO: this should at least be a dict
        self.multi_loss.losses[2].current_saliency_score = saliencies  # TODO: this should at least be a dict
        loss = self.multi_loss.train(sample) / self._settings.TRAINER.ACCUM_STEPS
        self.feature_match_recall.train(sample)
        loss.backward()
        return True

    @handle_OOM
    def _custom_val_step(self, sample: Pair) -> bool:
        sample.correspondences
        sample, overlaps, saliencies = self.model(sample)
        sample.source.first.pcd = sample.source.first.pcd.transform(sample.GT_tf_matrix)
        self.multi_loss.losses[1].current_overlap_score = overlaps
        self.multi_loss.losses[2].current_saliency_score = saliencies
        self.multi_loss.val(sample)
        self.feature_match_recall.val(sample)
        self.state.val.on_best_iter = self.multi_loss.val.on_best_iter
        return True

    def _custom_epoch_step(self) -> None:
        self.multi_loss.losses[-1].weight = 1.0 if self.feature_match_recall.val.get("average") > 0.3 else 0.0
        self.scheduler.step()
        log.info(f"Current learning rate: {self.scheduler.get_last_lr()}")

    def _custom_loss_log(self, mode: str) -> str:
        assert (mode := mode.lower()) in ["train", "val"]
        mode_color = "orange1" if mode == "train" else "bright_blue"
        return (
            f"[{mode_color}][{datetime.now().strftime('%H:%M:%S')}][/{mode_color}]"
            + f" [bold {mode_color}][{mode.upper()}][/bold {mode_color}] {getattr(self.state, mode).log_header}"
            + f" FMR[{getattr(self.feature_match_recall, mode).get('average'):5.4f}]"
            + f" {getattr(self.multi_loss, mode)}"
            + f" = {[getattr(loss, mode) for loss in self.multi_loss.losses]}"
        )
