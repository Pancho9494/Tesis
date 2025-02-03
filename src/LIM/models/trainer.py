from abc import ABC, abstractmethod
import aim
import copy
from dataclasses import dataclass, field
from functools import partial
import gc
import multiprocessing as mp
from pathlib import Path
import torch
from typing import Dict, Any, Union, Tuple, Callable

from config import settings
from LIM.data.structures import Cloud, transform_factory, Pair
from LIM.data.sets import CloudDatasetsI
from LIM.metrics import MultiLoss, Loss, L1Loss, IOU, CircleLoss, OverlapLoss, MatchabilityLoss
import functools


def handle_OOM(func: Callable) -> Callable:
    @functools.wraps(func)
    def inner(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except RuntimeError:
            return None


class BaseTrainer(ABC):
    @dataclass
    class RunState:
        @dataclass
        class Current:
            epoch: int = field(default=0)
            step: int = field(default=0)
            best_model: Dict[str, Any] = field(default_factory=dict)

        tracker: aim.Run = field(default_factory=aim.Run)
        current: Current = field(default_factory=Current)

        @property
        def on_first_step(self) -> bool:
            return self.current.step == 0

        @property
        def on_validation_step(self) -> bool:
            return self.current.step % settings.TRAINER.VALIDATION_PERIOD == 0

        @property
        def on_accumulation_step(self) -> bool:
            return (self.current.step + 1) % settings.TRAINER.ACCUM_STEPS == 0

        @property
        def on_backup_step(self) -> bool:
            return self.current.step % settings.TRAINER.BACKUP_PERIOD == 0

    device: torch.device = torch.device(settings.DEVICE)
    state: RunState
    model: torch.nn.Module
    dataloader: torch.utils.data.DataLoader
    dataset: CloudDatasetsI
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
    scaler: torch.amp.GradScaler

    def __init__(self, tag: str, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        self.tag = tag
        self.model = model.to(self.device)
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=settings.TRAINER.LEARNING_RATE,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=settings.TRAINER.EPOCHS,
            eta_min=1e-4,
        )
        self.scaler = torch.amp.GradScaler(settings.DEVICE)
        self.state = BaseTrainer.RunState()
        self.__make_split()

    @property
    def current(self) -> RunState.Current:
        return self.state.current

    def train(self) -> None:
        for self.current.epoch in range(self.current.epoch, settings.TRAINER.EPOCHS):
            for self.current.step, sample in enumerate(self.train_loader):
                self.__clean_memory()
                try:
                    self.__train_step(sample)
                    self.__val_step()
                except RuntimeError:
                    self.dataset.force_downsample(sample)
                    continue
                self.__log(show=True)
                self.__backup_model(self.model.state_dict(), "latest")
            self.scheduler.step()
            print(f"Current learning rate: {self.scheduler.get_last_lr()}")

    def __train_step(self, sample: Any) -> None:
        self.model.train()
        self._custom_train_step(sample)
        on_last_batch = (self.current.step + 1) == len(self.train_loader)
        if self.state.on_accumulation_step or on_last_batch:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    @abstractmethod
    def _custom_train_step(self, sample: Any) -> None: ...

    def __val_step(self) -> None:
        if self.state.on_first_step or not self.state.on_validation_step:
            return

        self.model.eval()
        with torch.no_grad():
            if _BEST_MODEL := self._custom_val_step():
                self.current.best_model = self.model.state_dict()
                self.__backup_model(self.current.best_model, "best")

    @abstractmethod
    def _custom_val_step() -> bool: ...

    def __clean_memory(self) -> None:
        """
        I'm not entirely sure if this is necessary, but it seems to help getting less cuda out of memory errors
        """
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def load_model(self, path: Union[str, Path]) -> None:
        print(f"Loading model from {path}")
        backup: Dict[str, Any] = torch.load(path, weights_only=False)
        self.tag = backup["tag"]
        self.model.load_state_dict(backup["model_state_dict"])
        self.optimizer.load_state_dict(backup["optimizer_state_dict"])
        self.current.step = backup["current_step"]
        self.current.epoch = backup["current_epoch"]

    def __backup_model(self, model: Dict[str, Any], name: str) -> None:
        if self.state.on_first_step or not self.state.on_backup_step:
            return

        torch.save(
            {
                "tag": self.tag,
                "model_state_dict": model,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "current_step": self.current.step,
                "current_epoch": self.current.epoch,
                "params": settings.__dict__,
            },
            f"./weights/{self.tag}_{name}.tar",
        )

    def __log(self, show: bool = False) -> None:
        if not show:
            return

        step_id = f"Epoch[{self.current.epoch:02d}]"
        step_id += f" Step[{self.current.step:02d}]"
        step_id += f" Iter[{(self.current.epoch * len(self.train_loader)) + (self.current.step):02d}]"
        print(step_id + self._custom_loss_log())

    def __make_split(self) -> None:
        train_set, val_set = torch.utils.data.random_split(
            self.dataset, [1 - settings.TRAINER.VALIDATION_SPLIT, settings.TRAINER.VALIDATION_SPLIT]
        )
        train_set.dataset, val_set.dataset = copy.deepcopy(self.dataset), copy.deepcopy(self.dataset)

        print(f"Train split has {len(train_set.indices)} point clouds")
        print(f"Validation split has {len(val_set.indices)} point clouds")

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=settings.TRAINER.BATCH_SIZE,
            shuffle=True,
            collate_fn=partial(
                self.dataset.collate_fn,
                tf_pipeline=transform_factory(settings.TRAINER.POINTCLOUD_TF.TRAIN),
                # cloud_tf=transform_factory(settings.TRAINER.POINTCLOUD_TF.TRAIN),
                # implicit_tf=transform_factory(settings.TRAINER.IMPLICIT_GRID_TF.TRAIN),
            ),
            num_workers=mp.cpu_count() - 4 if settings.TRAINER.MULTIPROCESSING else 0,
            multiprocessing_context="spawn" if settings.TRAINER.MULTIPROCESSING else None,
            pin_memory=True,
        )
        self.val_loader = iter(
            torch.utils.data.DataLoader(
                dataset=val_set,
                batch_size=1,
                shuffle=False,
                collate_fn=partial(
                    self.dataset.collate_fn,
                    tf_pipeline=transform_factory(settings.TRAINER.POINTCLOUD_TF.TRAIN),
                    # cloud_tf=transform_factory(settings.TRAINER.POINTCLOUD_TF.VALIDATION),
                    # implicit_tf=transform_factory(settings.TRAINER.IMPLICIT_GRID_TF.VALIDATION),
                ),
                num_workers=2 if settings.TRAINER.MULTIPROCESSING else 0,
                multiprocessing_context="spawn" if settings.TRAINER.MULTIPROCESSING else None,
                pin_memory=True,
            )
        )

    @abstractmethod
    def _custom_loss_log(self) -> str: ...


class IAETrainer(BaseTrainer):
    l1_loss: Loss
    iou_loss: Loss

    def __init__(self, tag: str, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        super(IAETrainer, self).__init__(tag, model, dataset)
        self.l1_loss = L1Loss(trainer_state=self.state, reduction="none")
        self.iou_loss = IOU(trainer_state=self.state, threshold=0.5)

    def _custom_train_step(self, sample: Tuple[Cloud, Cloud]) -> None:
        cloud, implicit = sample
        with torch.amp.autocast(settings.DEVICE):
            predicted_df = self.model(cloud, implicit)
            self.l1_loss.train(sample=(predicted_df, implicit.features))
            self.iou_loss.train(sample=(predicted_df, implicit.features))

    def _custom_val_step(self) -> bool:
        cloud, implicit = next(self.val_loader)
        predicted_df = self.model(cloud, implicit)
        self.l1_loss.val(sample=(predicted_df, implicit.features))
        self.iou_loss.val(sample=(predicted_df, implicit.features))
        return self.l1_loss.val.on_best_iter

    def _custom_loss_log(self) -> str:
        out = f"\tL1LossTrain[{self.l1_loss.train.current:5.2f}] IOUTrain[{self.iou_loss.train.current:5.2f}]"
        out += (
            f"\tL1LossVal[{self.l1_loss.val.current:5.2f}] IOUVal[{self.iou_loss.val.current:5.2f}]"
            if not self.state.on_first_step and self.state.on_validation_step
            else ""
        )
        return out


class PredatorTrainer(BaseTrainer):
    circle_loss: CircleLoss
    overlap_loss: OverlapLoss
    matchability_loss: MatchabilityLoss
    multi_loss: MultiLoss

    def __init__(self, tag: str, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        super(PredatorTrainer, self).__init__(tag, model, dataset)
        self.circle_loss = CircleLoss(trainer_state=self.state, weight=1.0)
        self.overlap_loss = OverlapLoss(trainer_state=self.state, weight=1.0)
        self.matchability_loss = MatchabilityLoss(trainer_state=self.state, weight=0.0)
        self.multi_loss = MultiLoss(
            losses=[
                self.circle_loss,
                self.overlap_loss,
                self.matchability_loss,
            ],
        )

    def _custom_train_step(self, sample: Pair) -> None:
        with torch.no_grad():
            sample.correspondences
            sample.source.pcd = sample.source.pcd.transform(sample.GT_tf_matrix)

        loss = self.multi_loss.train(self.model(sample)) / settings.TRAINER.ACCUM_STEPS
        self.scaler.scale(loss).backward(retain_graph=True)

    def _custom_val_step(self) -> bool:
        self.multi_loss.val(self.model(next(self.val_loader)))
        return self.multi_loss.val.on_best_iter

    def _custom_loss_log(self) -> str:
        out = f"\tMultiTrain[{self.multi_loss.train.current:5.4f}] = "
        out += f"Circle[{self.circle_loss.train.current:5.4f}] + "
        out += f"Overlap[{self.overlap_loss.train.current:5.4f}] + "
        out += f"Match[{self.matchability_loss.train.current:5.4f}]"

        if not self.state.on_first_step and self.state.on_validation_step:
            out += f"\tMultiVal[{self.multi_loss.val.current:5.4f}]  = "
            out += f"Circle[{self.circle_loss.val.current:5.4f}] + "
            out += f"Overlap[{self.overlap_loss.val.current:5.4f}] + "
            out += f"Match[{self.matchability_loss.val.current:5.4f}]"

        return out
