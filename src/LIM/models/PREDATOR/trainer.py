import torch
from config.config import settings
from LIM.models.trainer import BaseTrainer, handle_OOM
from LIM.metrics import MultiLoss, FeatureMatchRecall, CircleLoss, OverlapLoss, MatchabilityLoss
from LIM.data.sets import CloudDatasetsI
from LIM.data.structures import Pair


class PredatorTrainer(BaseTrainer):
    multi_loss: MultiLoss
    feature_match_recall: FeatureMatchRecall

    def __init__(self, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        super(PredatorTrainer, self).__init__(model, dataset)
        self.multi_loss = MultiLoss(
            losses=[
                CircleLoss(trainer_state=self.state, weight=1.0),
                OverlapLoss(trainer_state=self.state, weight=1.0),
                MatchabilityLoss(trainer_state=self.state, weight=0.0),
            ],
        )
        self.feature_match_recall = FeatureMatchRecall(trainer_state=self.state)

    # @handle_OOM
    def _custom_train_step(self, sample: Pair) -> bool:
        sample.correspondences
        sample, overlaps, saliencies = self.model(sample)
        sample.source.first.pcd = sample.source.first.pcd.transform(sample.GT_tf_matrix)
        self.multi_loss.losses[1].current_overlap_score = overlaps  # TODO: this should at least be a dict
        self.multi_loss.losses[2].current_saliency_score = saliencies  # TODO: this should at least be a dict
        loss = self.multi_loss.train(sample) / settings.TRAINER.ACCUM_STEPS
        self.feature_match_recall.train(sample)
        loss.backward()
        return True

    # @handle_OOM
    def _custom_val_step(self, sample: Pair) -> bool:
        sample.correspondences
        sample = self.model(sample)
        sample.source.first.pcd = sample.source.first.pcd.transform(sample.GT_tf_matrix)
        self.multi_loss.val(sample)
        self.feature_match_recall.val(sample)
        self.state.val.on_best_iter = self.multi_loss.val.on_best_iter
        return True

    def _custom_epoch_step(self) -> None:
        self.multi_loss.losses[-1].weight = 1.0 if self.feature_match_recall.val.get("average") > 0.3 else 0.0

    def _custom_loss_log(self, mode: str) -> None:
        assert (mode := mode.lower()) in ["train", "val"]
        print(
            f"{mode.upper()}:\t{getattr(self.state, mode).log_header}"
            + f" FMR[{getattr(self.feature_match_recall, mode).get('average'):5.4f}]"
            + f" {getattr(self.multi_loss, mode)}"
            + f" = {[getattr(loss, mode) for loss in self.multi_loss.losses]}"
        )
