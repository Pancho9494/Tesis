from abc import ABC, abstractmethod
from aim import Run
import copy
from dataclasses import dataclass, field
from functools import partial
import gc
import multiprocessing as mp
from pathlib import Path
import torch
import torchvision
from typing import Dict, List, Any, Union, Tuple

from config import settings
from LIM.data.structures import Cloud, transform_factory, Pair
from LIM.data.sets import CloudDatasetsI, collate_scannet
from LIM.metrics.losses import Loss, L1Loss, IOU, CircleLoss, OverlapLoss, MatchabilityLoss, MultiLoss

from debug.context import inspect_tensor

torch.multiprocessing.set_start_method("spawn", force=True)


class Transforms:
    cloud_tf: torchvision.transforms.Compose
    implicit_tf: torchvision.transforms.Compose

    def __init__(self, cloud_tf: List[torch.nn.Module], implicit_tf: List[torch.nn.Module]) -> None:
        self.cloud_tf = torchvision.transforms.Compose(cloud_tf)
        self.implicit_tf = torchvision.transforms.Compose(implicit_tf)


class BaseTrainer(ABC):
    @dataclass
    class Current:
        epoch: int = field(default=0)
        step: int = field(default=0)
        best_model: Dict[str, Any] = field(default_factory=dict)

    device: torch.device = torch.device(settings.DEVICE)
    run: Run
    current: Current
    model: torch.nn.Module
    dataloader: torch.utils.data.DataLoader
    dataset: CloudDatasetsI
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
    scaler: torch.amp.GradScaler
    transforms: Dict[str, Transforms]

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

        self.__make_split()
        self.current = BaseTrainer.Current()
        self.run = Run()
        # self.run["trainer"] = settings.TRAINER.__dict__
        # self.run["model"] = settings.MODEL.__dict__

    def train(self) -> None:
        for self.current.epoch in range(self.current.epoch, settings.TRAINER.EPOCHS):
            for self.current.step, sample in enumerate(self.train_loader, self.current.step):
                self.__clean_memory()
                self.__train_step(sample)
                self.__val_step()
                self.__log(show=True)
                self.__backup_model(self.model.state_dict(), "latest")
            self.scheduler.step()
            print(f"Current learning rate: {self.scheduler.get_last_lr()}")

    def __train_step(self, sample: Any) -> None:
        self.model.train()

        self._custom_train_step(sample)

        accumulation_done = (self.current.step + 1) % settings.TRAINER.ACCUM_STEPS == 0
        on_last_batch = (self.current.step + 1) == len(self.train_loader)
        if accumulation_done or on_last_batch:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # torch.cuda.memory._dump_snapshot("src/debug/profiling/full.pickle")

    @abstractmethod
    def _custom_train_step(self, sample: Any) -> None: ...

    def __val_step(self) -> None:
        ON_FIRST_STEP: bool = self.current.step == 0
        ON_VALIDATION_STEP: bool = self.current.step % settings.TRAINER.VALIDATION_PERIOD == 0
        if ON_FIRST_STEP or not ON_VALIDATION_STEP:
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
        ON_FIRST_STEP: bool = self.current.step == 0
        ON_BACKUP_STEP: bool = self.current.step % settings.TRAINER.BACKUP_PERIOD == 0
        if ON_FIRST_STEP or not ON_BACKUP_STEP:
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

        step_id = f"Epoch[{(self.current.epoch):02d}]\tIter[{(self.current.step):02d}]"
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
        self.l1_loss = L1Loss(scaler=self.scaler, reduction="none", accum_steps=settings.TRAINER.ACCUM_STEPS)
        self.iou_loss = IOU(scaler=self.scaler, threshold=0.5)

    def _custom_train_step(self, sample: Tuple[Cloud, Cloud]) -> None:
        cloud, implicit = sample
        with torch.amp.autocast(settings.DEVICE):
            predicted_df = self.model(cloud, implicit)
            self.l1_loss.train(predicted_df, implicit.features, with_grad=True)
            self.iou_loss.train(predicted_df, implicit.features)

    def _custom_val_step(self) -> bool:
        cloud, implicit = next(self.val_loader)
        predicted_df = self.model(cloud, implicit)
        _best_l1_loss: bool = self.l1_loss.val(predicted_df, implicit.features)
        best_iou: bool = self.iou_loss.val(predicted_df, implicit.features)
        return best_iou

    def _custom_loss_log(self) -> str:
        ON_FIRST_STEP: bool = self.current == 0
        ON_VALIDATION_STEP: bool = self.current.step % settings.TRAINER.VALIDATION_PERIOD == 0
        out = f"\tL1LossTrain[{self.l1_loss.get('train'):5.2f}] IOUTrain[{self.iou_loss.get('train'):5.2f}]"
        out += (
            f"\tL1LossVal[{self.l1_loss.get('val'):5.2f}] IOUVal[{self.iou_loss.get('val'):5.2f}]"
            if not ON_FIRST_STEP and ON_VALIDATION_STEP
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
        self.circle_loss = CircleLoss(scaler=self.scaler, weight=1.0)
        self.overlap_loss = OverlapLoss(scaler=self.scaler, weight=1.0)
        self.matchability_loss = MatchabilityLoss(scaler=self.scaler, weight=0.0)
        self.multi_loss = MultiLoss(
            losses=[
                self.circle_loss,
                self.overlap_loss,
                self.matchability_loss,
            ],
            scaler=self.scaler,
        )

    def _custom_train_step(self, sample: Pair) -> None:
        prediction = self.model(sample)
        self.multi_loss.train(prediction, with_grad=True)

    def _custom_val_step(self) -> bool:
        sample: Pair = next(self.val_loader)
        prediction = self.model(sample)
        best_multi_loss: bool = self.multi_loss.val(prediction)

        return best_multi_loss

    def _custom_loss_log(self):
        ON_FIRST_STEP: bool = self.current == 0
        ON_VALIDATION_STEP: bool = self.current.step % settings.TRAINER.VALIDATION_PERIOD == 0
        out = f"\tMultiLossTrain[{self.multi_loss.get('train'):5.4f}] = "
        out += f"Circle[{self.circle_loss.get('train'):5.4f}] + "
        out += f"Overlap[{self.overlap_loss.get('train'):5.4f}] + "
        out += f"Match[{self.matchability_loss.get('train'):5.4f}]"

        if not ON_FIRST_STEP and ON_VALIDATION_STEP:
            out += f"\tMultiLossVal[{self.multi_loss.get('val'):5.4f}]  = "
            out += f"Circle[{self.circle_loss.get('val'):5.4f}] + "
            out += f"Overlap[{self.overlap_loss.get('val'):5.4f}] + "
            out += f"Match[{self.matchability_loss.get('val'):5.4f}]"

        self.run.track(
            self.multi_loss.get("train"),
            name="MultiLoss",
            step=self.current.step,
            epoch=self.current.epoch,
            context={"subset": "train"},
        )
        self.run.track(
            self.circle_loss.get("train"),
            name="CircleLoss",
            step=self.current.step,
            epoch=self.current.epoch,
            context={"subset": "train"},
        )
        self.run.track(
            self.overlap_loss.get("train"),
            name="OverlapLoss",
            step=self.current.step,
            epoch=self.current.epoch,
            context={"subset": "train"},
        )
        self.run.track(
            self.matchability_loss.get("train"),
            name="MatchabilityLoss",
            step=self.current.step,
            epoch=self.current.epoch,
            context={"subset": "train"},
        )

        self.run.track(
            self.multi_loss.get("val"),
            name="MultiLoss",
            step=self.current.step,
            epoch=self.current.epoch,
            context={"subset": "val"},
        )
        self.run.track(
            self.circle_loss.get("val"),
            name="CircleLoss",
            step=self.current.step,
            epoch=self.current.epoch,
            context={"subset": "val"},
        )
        self.run.track(
            self.overlap_loss.get("val"),
            name="OverlapLoss",
            step=self.current.step,
            epoch=self.current.epoch,
            context={"subset": "val"},
        )
        self.run.track(
            self.matchability_loss.get("val"),
            name="MatchabilityLoss",
            step=self.current.step,
            epoch=self.current.epoch,
            context={"subset": "val"},
        )
        return out
