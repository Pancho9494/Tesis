import torch
from typing import Tuple

from config.config import settings
from LIM.models.trainer import BaseTrainer, handle_OOM
from LIM.metrics import L1Loss, IOU
from LIM.data.sets import CloudDatasetsI
from LIM.data.structures import PCloud
import random
import numpy as np
import copy


class IAETrainer(BaseTrainer):
    l1_loss: L1Loss
    iou_loss: IOU

    def __init__(self, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        super(IAETrainer, self).__init__(model, dataset)
        self.l1_loss = L1Loss(trainer_state=self.state, reduction="none")
        self.iou_loss = IOU(trainer_state=self.state, threshold=0.5)

    @handle_OOM
    def _custom_train_step(self, sample: Tuple[PCloud, PCloud]) -> bool:
        og_points = copy.copy(sample[0])
        cloud, implicit = sample
        predicted_df = self.model(cloud, implicit)
        loss = self.l1_loss.train(sample=(predicted_df, implicit.features)) / settings.TRAINER.ACCUM_STEPS
        loss.backward()

        # TODO: I don't like doing this like this but I'm running out of time
        random_points_iou_file = np.load(str(random.choice(list(implicit.path[0].parent.iterdir()))))
        implicit = PCloud.from_arr(random_points_iou_file["points"].astype(np.float32))
        implicit.features = np.expand_dims(random_points_iou_file["df_value"].astype(np.float32), axis=1)
        predicted_df = self.model(og_points, implicit)
        self.iou_loss.train(sample=(predicted_df, implicit.features))
        return True

    @handle_OOM
    def _custom_val_step(self) -> bool:
        try:
            cloud, implicit = next(self.val_loader)
        except StopIteration:
            self.val_loader = iter(
                self.make_dataloader(self.val_set, num_workers=2 if settings.TRAINER.MULTIPROCESSING else 0)
            )
            cloud, implicit = next(self.val_loader)

        og_points = copy.copy(cloud)
        predicted_df = self.model(cloud, implicit)
        self.l1_loss.val(sample=(predicted_df, implicit.features))
        # TODO: I don't like doing this like this but I'm running out of time
        random_points_iou_file = np.load(str(random.choice(list(implicit.path[0].parent.iterdir()))))
        implicit = PCloud.from_arr(random_points_iou_file["points"].astype(np.float32))
        implicit.features = np.expand_dims(random_points_iou_file["df_value"].astype(np.float32), axis=1)
        predicted_df = self.model(og_points, implicit)
        self.iou_loss.val(sample=(predicted_df, implicit.features))
        return self.iou_loss.val.on_best_iter

    def _custom_epoch_step(self) -> None: ...

    def _custom_loss_log(self, mode: str) -> None:
        assert (mode := mode.lower()) in ["train", "val"]
        print(
            f"{mode.upper()}:\t{getattr(self.state, mode).log_header}"
            + f" {getattr(self.l1_loss, mode)}"
            + f" {getattr(self.iou_loss, mode)}"
        )
