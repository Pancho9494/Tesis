from abc import ABC, abstractmethod
import aim
from dataclasses import dataclass, field
from functools import partial
import gc
import multiprocessing as mp
import torch
from typing import Dict, Any, Callable, List, Optional
import torch.utils.data.dataloader
import functools

from config import settings
from LIM.data.sets import CloudDatasetsI


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

    @abstractmethod
    def _custom_train_step(self, sample: Any) -> bool: ...

    @abstractmethod
    def _custom_val_step(self, sample: Any) -> bool: ...

    @abstractmethod
    def _custom_epoch_step() -> None: ...

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
                _ON_LAST_BATCH := (self.state.train.step + 1) == len(self.train_loader)
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.state.on_backup_step:
                self.model.save("latest")

            self._custom_loss_log(mode="train")
            self.state.train.iteration += 1
        return True

    def __val_step(self) -> bool:
        self.model.eval()
        with torch.no_grad():
            for self.state.val.step, sample in enumerate(self.val_loader):
                self.__clean_memory()
                if not self._custom_val_step(sample):
                    self.val_set.force_downsample(sample)

                if self.state.val.on_best_iter:
                    self.model.backup("best")
                    self.state.val.on_best_iter = False

                self._custom_loss_log(mode="val")
                self.state.val.iteration += 1
        return True

    def __clean_memory(self) -> None:
        if self.device != torch.device("cuda"):
            return
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

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
    def _custom_loss_log(self, mode: str) -> None: ...
