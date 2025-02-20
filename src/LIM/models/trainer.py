from abc import ABC, abstractmethod
import aim
import copy
from dataclasses import dataclass, field
from functools import partial
import gc
import multiprocessing as mp
from pathlib import Path
import torch
from typing import Dict, Any, Union, Tuple, Callable, List, Optional

import torch.utils.data.dataloader

from config import settings
from LIM.data.structures import PCloud, Pair
from LIM.data.sets import CloudDatasetsI
from LIM.metrics import (
    MultiLoss,
    Loss,
    L1Loss,
    IOU,
    CircleLoss,
    OverlapLoss,
    MatchabilityLoss,
    FeatureMatchRecall,
)
import functools


def handle_OOM(func: Callable) -> bool:
    @functools.wraps(func)
    def inner(*args, **kwargs) -> Any:
        try:
            func(*args, **kwargs)
            return True
        except RuntimeError:
            print("Cuda OOM! ", end="")
            return False

    return inner


@dataclass
class RunState:
    @dataclass
    class Current:
        iteration: int = field(default=0)
        epoch: int = field(default=0)
        setp: int = field(default=0)
        best_model: Dict[str, Any] = field(default_factory=dict)
        on_best_iter: bool = False

        @property
        def log_header(self) -> str:
            return f"Epoch[{self.epoch:02d}]" + f"Step[{self.step:02d}]" + f"Iter[{self.iteration:02d}]"

    train: Current = field(default_factory=Current)
    val: Current = field(default_factory=Current)
    tracker: aim.Run = field(default_factory=aim.Run)

    @property
    def on_first_step(self) -> bool:
        return self.train.step == 0

    @property
    def on_accumulation_step(self) -> bool:
        return (self.train.step + 1) % settings.TRAINER.ACCUM_STEPS == 0

    @property
    def on_backup_step(self) -> bool:
        return self.train.step % settings.TRAINER.BACKUP_PERIOD == 0


class BaseTrainer(ABC):
    device: torch.device = torch.device(settings.DEVICE)
    state: RunState
    model: torch.nn.Module
    dataloader: torch.utils.data.DataLoader
    dataset: CloudDatasetsI
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler

    def __init__(self, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        self.model = model.to(self.device)
        self.dataset = dataset
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=settings.TRAINER.LEARNING_RATE,
            weight_decay=settings.TRAINER.WEIGHT_DECAY,
            momentum=settings.TRAINER.MOMENTUM,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        self.state = RunState()
        self.train_set, self.val_set = self.dataset.train_set(), self.dataset.val_set()

        self.train_loader = self.make_dataloader(
            self.train_set,
            num_workers=mp.cpu_count() - 4 if settings.TRAINER.MULTIPROCESSING else 0,
        )
        self.val_loader = self.make_dataloader(
            self.val_set,
            num_workers=2 if settings.TRAINER.MULTIPROCESSING else 0,
        )

        print(f"Training set has {len(self.train_set)} samples, validation set has {len(self.val_set)} samples")

    def __repr__(self) -> str:
        return self.__clas__.__name__

    def train(self) -> None:
        for self.state.train.epoch in range(self.state.train.epoch, settings.TRAINER.EPOCHS):
            self.state.val.epoch = self.state.train.epoch
            self.optimizer.zero_grad()
            self.__clean_memory()
            self.__train_step()
            self.__val_step()
            self._custom_epoch_step()
            self.scheduler.step()
            print(f"Current learning rate: {self.scheduler.get_last_lr()}")

    def __train_step(self) -> bool:
        self.model.train()
        for self.state.train.step, sample in enumerate(self.train_loader):
            self.__clean_memory()
            if not self._custom_train_step(sample):
                self.train_set.force_downsample(sample)

            if self.state.on_accumulation_step or (
                _on_last_batch := (self.state.train.step + 1) == len(self.train_loader)
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.__backup_model(self.model.state_dict(), "latest")

            print(
                f"TRAIN:\t{self.state.train.log_header}"
                + f" FMR[{self.feature_match_recall.train.get('average'):5.4f}]"
                + f" {self.multi_loss.train} "
                + f"({self.overlap_loss.train} + [{self.match_loss.weight}]{self.match_loss.train} + {self.circle_loss.train})"
            )

            self.state.train.iteration += 1
        return True

    @abstractmethod
    def _custom_train_step(self, sample: Any) -> None: ...

    def __val_step(self) -> bool:
        self.model.eval()
        with torch.no_grad():
            for self.state.val.step, sample in enumerate(self.val_loader):
                self.__clean_memory()
                if not self._custom_val_step(sample):
                    self.val_set.force_downsample(sample)

                if self.state.val.on_best_iter:
                    self.state.val.best_model = self.model.state_dict()
                    self.__backup_model(self.state.val.best_model, "best")
                    self.state.val.on_best_iter = False

                print(
                    f"VAL:\t{self.state.val.log_header}"
                    + f" FMR[{self.feature_match_recall.val.get('average'):5.4f}]"
                    + f" {self.multi_loss.val} "
                    + f"({self.overlap_loss.val} + [{self.match_loss.weight}]{self.match_loss.val} + {self.circle_loss.val})"
                )
                self.state.val.iteration += 1
        return True

    @abstractmethod
    def _custom_val_step(self, sample: Any) -> bool: ...

    @abstractmethod
    def _custom_epoch_step() -> None: ...

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
        self.model.load_state_dict(backup["model_state_dict"])
        self.optimizer.load_state_dict(backup["optimizer_state_dict"])
        self.state.train.train_step = backup["current_step"]
        self.state.train.epoch = backup["current_epoch"]

    def __backup_model(self, model: Dict[str, Any], name: str) -> None:
        if self.state.on_first_step or not self.state.on_backup_step:
            return

        torch.save(
            {
                "model_state_dict": model,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "current_step": self.state.train.step,
                "current_epoch": self.state.train.epoch,
                "params": settings.__dict__,
            },
            f"./weights/{self.__class__.__name__}_{name}.tar",
        )

    def make_dataloader(
        self,
        dataset: CloudDatasetsI,
        num_workers: int,
        tf_pipeline: Optional[List[torch.nn.Module]] = None,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=settings.TRAINER.BATCH_SIZE,
            shuffle=True,
            collate_fn=partial(
                dataset.collate_fn,
                tf_pipeline=tf_pipeline,
            ),
            num_workers=num_workers,
            multiprocessing_context="spawn" if settings.TRAINER.MULTIPROCESSING else None,
            persistent_workers=True if settings.TRAINER.MULTIPROCESSING else False,
        )

    @abstractmethod
    def _custom_loss_log(self) -> str: ...


class IAETrainer(BaseTrainer):
    l1_loss: Loss
    iou_loss: Loss

    def __init__(self, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        super(IAETrainer, self).__init__(model, dataset)
        self.l1_loss = L1Loss(trainer_state=self.state, reduction="none")
        self.iou_loss = IOU(trainer_state=self.state, threshold=0.5)

    def _custom_train_step(self, sample: Tuple[PCloud, PCloud]) -> None:
        cloud, implicit = sample
        with torch.amp.autocast(settings.DEVICE):
            predicted_df = self.model(cloud, implicit)
            self.l1_loss.train(sample=(predicted_df, implicit.features))
            self.iou_loss.train(sample=(predicted_df, implicit.features))

    def _custom_val_step(self) -> bool:
        try:
            cloud, implicit = next(self.val_loader)
        except StopIteration:
            self.val_loader = iter(
                self.make_dataloader(self.val_set, num_workers=2 if settings.TRAINER.MULTIPROCESSING else 0)
            )
            cloud, implicit = next(self.val_loader)

        predicted_df = self.model(cloud, implicit)
        self.l1_loss.val(sample=(predicted_df, implicit.features))
        self.iou_loss.val(sample=(predicted_df, implicit.features))
        return self.l1_loss.val.on_best_iter

    def _custom_epoch_step(self) -> None: ...

    def _custom_loss_log(self) -> str:
        return ""


class PredatorTrainer(BaseTrainer):
    overlap_loss: OverlapLoss
    match_loss: MatchabilityLoss
    circle_loss: CircleLoss
    multi_loss: MultiLoss
    feature_match_recall: FeatureMatchRecall

    @dataclass
    class AverageMeter:
        val: float = field(default=0.0)
        avg: float = field(default=0.0)
        sum: float = field(default=0.0)
        sq_sum: float = field(default=0.0)
        count: int = field(default=0)

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0.0
            self.sq_sum = 0.0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            self.sq_sum += val**2 * n
            self.var = self.sq_sum / self.count - self.avg**2

    def __init__(self, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        super(PredatorTrainer, self).__init__(model, dataset)
        self.overlap_loss = OverlapLoss(trainer_state=self.state, weight=1.0)
        self.match_loss = MatchabilityLoss(trainer_state=self.state, weight=0.0)
        self.circle_loss = CircleLoss(trainer_state=self.state, weight=1.0)
        self.multi_loss = MultiLoss(
            losses=[
                self.circle_loss,
                self.overlap_loss,
                self.match_loss,
            ],
        )
        self.feature_match_recall = FeatureMatchRecall(trainer_state=self.state)

    @handle_OOM
    def _custom_train_step(self, sample: Pair) -> bool:
        sample.correspondences
        sample = self.model(sample)
        sample.source.first.pcd = sample.source.first.pcd.transform(sample.GT_tf_matrix)
        loss = self.multi_loss.train(sample) / settings.TRAINER.ACCUM_STEPS
        self.feature_match_recall.train(sample)
        loss.backward()
        return True

    @handle_OOM
    def _custom_val_step(self, sample: Pair) -> bool:
        sample.correspondences
        sample = self.model(sample)
        sample.source.first.pcd = sample.source.first.pcd.transform(sample.GT_tf_matrix)
        self.multi_loss.val(sample)
        self.feature_match_recall.val(sample)
        self.state.val.on_best_iter = self.multi_loss.val.on_best_iter
        return True

    def _custom_epoch_step(self) -> None:
        self.match_loss.weight = 1.0 if self.feature_match_recall.val.get("average") > 0.3 else 0.0

    def _custom_loss_log(self) -> str:
        return ""
