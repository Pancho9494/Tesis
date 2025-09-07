from copy import copy
from datetime import datetime
from typing import Type

import torch

import config.config as config
from LIM.data.sets import CloudDatasetsI
from LIM.data.structures.pair import Pair
from LIM.data.structures.pcloud import Downsampler
from LIM.log import log
from LIM.metrics import CircleLoss, FeatureMatchRecall, MatchabilityLoss, MultiLoss, OverlapLoss
from LIM.metrics.losses import RRE, RTE
from LIM.models.evaluator import RANSAC
from LIM.models.modelI import Model
from LIM.training.trainer import BaseTrainer, handle_OOM


class PredatorTrainer(BaseTrainer):
    multi_loss: MultiLoss
    feature_match_recall: FeatureMatchRecall
    RRE: RRE
    RTE: RTE
    ransac: RANSAC
    downsampler: Downsampler
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
        self.losses.extend(self.multi_loss.losses + [self.feature_match_recall])

        self.RTE = RTE(trainer_state=self.state)
        self.RRE = RRE(trainer_state=self.state)

        if self._settings.TRAINER.MODE.value in [BaseTrainer.Mode.FIXED.value, BaseTrainer.Mode.LATEST.value]:
            for loss in self.losses:
                loss.load(run=self.BACKUP_DIR, suffix="latest")

        self.ransac = RANSAC(
            distance_threshold=self._settings.TESTER.RANSAC.DISTANCE_THRESHOLD,
            similarity_threshold=self._settings.TESTER.RANSAC.SIMILARITY_THRESHOLD,
            max_iterations=self._settings.TESTER.RANSAC.MAX_ITERATIONS,
            n_correspondences=3,
        )
        self.downsampler = Downsampler(size=500, mode=Downsampler.Mode.PROBABILISTIC)

    def _load_model(self, model: Type[Model]) -> None:
        self.model = model()
        self.model.to(self.device)

    @handle_OOM
    def _custom_train_step(self, sample: Pair) -> bool:
        if config.settings.MODEL.ENCODER.FREEZE:
            for block in self.model.encoder.frozen_blocks:
                block.eval()
        sample.correspondences
        sample, overlaps, saliencies = self.model(sample)
        aligned = copy(sample)
        aligned.source.first.pcd = aligned.source.first.pcd.transform(aligned.GT_tf_matrix)

        # TODO: these two should at least be a dict
        self.multi_loss.losses[1].current_overlap_score = overlaps
        self.multi_loss.losses[2].current_saliency_score = saliencies

        loss = self.multi_loss.train(aligned) / self._settings.TRAINER.ACCUM_STEPS
        self.feature_match_recall.train(aligned)
        loss.backward()

        downsampled = copy(sample)
        N = downsampled.source.shape[0]
        src_overlaps, src_saliencies = overlaps[:N], saliencies[:N]
        tgt_overlaps, tgt_saliencies = overlaps[N:], saliencies[N:]
        src_down, src_idxs = self.downsampler(downsampled.source, scores=src_overlaps * src_saliencies)
        tgt_down, tgt_idxs = self.downsampler(downsampled.target, scores=tgt_overlaps * tgt_saliencies)

        ransac_result = self.ransac(src_down, tgt_down)
        tf_matrix, pred_correspondences = ransac_result.transformation, ransac_result.correspondence_set

        rre = self.RRE.train((downsampled.GT_tf_matrix, tf_matrix))
        rte = self.RTE.train((downsampled.GT_tf_matrix, tf_matrix))

        if rre < 0 and rte < 0 or (self.state.train.step % 400 == 0):
            log.info(f"{tf_matrix=}")
            log.info(f"{type(overlaps)=}")
            sample.show(tf_matrix, pred_correspondences, src_idxs, tgt_idxs, overlaps)
            # self.model.show_tsne(sample)
            # self.model.show_umap(sample)
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

        N = sample.source.shape[0]
        src_overlaps, src_saliencies = overlaps[:N], saliencies[:N]
        tgt_overlaps, tgt_saliencies = overlaps[N:], saliencies[N:]

        src_down, src_idxs = self.downsampler(sample.source, scores=src_overlaps * src_saliencies)
        tgt_down, tgt_idxs = self.downsampler(sample.target, scores=tgt_overlaps * tgt_saliencies)

        ransac_result = self.ransac(src_down, tgt_down)
        tf_matrix, pred_correspondences = ransac_result.transformation, ransac_result.correspondence_set

        self.RRE.val((sample.GT_tf_matrix, tf_matrix))
        self.RTE.val((sample.GT_tf_matrix, tf_matrix))

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
            + f" [bold blue][{self.state.tracker.name.split(' ')[-1]}][/bold blue]"
            + f" [bold {mode_color}][{mode.upper()}][/bold {mode_color}] {getattr(self.state, mode).log_header}"
            + f" FMR[{getattr(self.feature_match_recall, mode).get('average'):5.4f}]"
            + f" RRE[{getattr(self.RRE, mode).get('average'):5.4f}]"
            + f" RTE[{getattr(self.RTE, mode).get('average'):5.4f}]"
            + f" {getattr(self.multi_loss, mode)}"
            + f" = {[getattr(loss, mode) for loss in self.multi_loss.losses]}"
        )
