import torch
from typing import Tuple, Optional
import LIM.log as log
from config.config import settings
from LIM.training.trainer import BaseTrainer, handle_OOM
from LIM.metrics import L1Loss, IOU
from LIM.data.sets import CloudDatasetsI
from LIM.data.structures import PCloud
import random
import numpy as np
import copy
from datetime import datetime
from LIM.models.modelI import Model
from LIM.models.IAE import IAE


class IAETrainer(BaseTrainer):
    l1_loss: L1Loss
    iou_loss: IOU
    average_val_iou: float
    best_average_val_iou: float

    def __init__(
        self, model: torch.nn.Module, dataset: CloudDatasetsI, mode: Optional[BaseTrainer.Mode] = None
    ) -> None:
        super(IAETrainer, self).__init__(model, dataset, mode)
        self.l1_loss = L1Loss(trainer_state=self.state, reduction="none")
        self.iou_loss = IOU(trainer_state=self.state, threshold=0.5)
        self.model.to(self.device)
        self.average_val_iou, self.best_average_val_iou = 0.0, 0.0

    def _load_model(self, model: Model) -> None:
        self.model = IAE(model)

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
    def _custom_val_step(self, sample: Tuple[PCloud, PCloud]) -> bool:
        og_points = copy.copy(sample[0])
        cloud, implicit = sample
        predicted_df = self.model(cloud, implicit)
        self.l1_loss.val(sample=(predicted_df, implicit.features))
        # TODO: I don't like doing this like this but I'm running out of time
        random_points_iou_file = np.load(str(random.choice(list(implicit.path[0].parent.iterdir()))))
        implicit = PCloud.from_arr(random_points_iou_file["points"].astype(np.float32))
        implicit.features = np.expand_dims(random_points_iou_file["df_value"].astype(np.float32), axis=1)
        predicted_df = self.model(og_points, implicit)
        self.iou_loss.val(sample=(predicted_df, implicit.features))
        self.state.val.on_best_iter = self.iou_loss.val.on_best_iter
        self.average_val_iou += self.iou_loss.val.current
        return True

    def _custom_epoch_step(self) -> None:
        """
        In the original IAE model they decide the best model based on the mean IOU over all the validation set
        """
        self.average_val_iou /= self.state.val.step
        if self.average_val_iou > self.best_average_val_iou:
            log.info(f"New best average model with IOU={self.average_val_iou:2.4f}, making backup")
            self.best_average_val_iou = self.average_val_iou
            self.model.save_async(run=self.BACKUP_DIR, suffix="best_average")
            self.state.save_async(run=self.BACKUP_DIR, suffix="best_average")
        self.average_val_iou = 0

    def _custom_loss_log(self, mode: str) -> str:
        assert (mode := mode.lower()) in ["train", "val"]
        mode_color = "orange1" if mode == "train" else "bright_blue"
        return (
            f"[{mode_color}][{datetime.now().strftime('%H:%M:%S')}][/{mode_color}]"
            + f" [bold {mode_color}][{mode.upper()}][/bold {mode_color}] {getattr(self.state, mode).log_header}"
            + f" {getattr(self.l1_loss, mode)}"
            + f" {getattr(self.iou_loss, mode)}"
        )
