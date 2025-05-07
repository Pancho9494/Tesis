from abc import ABC, abstractmethod
from functools import partial
import gc
import multiprocessing as mp
import torch
from typing import Any, Callable, Optional, Tuple
import functools
import config.config as config
from LIM.data.sets import CloudDatasetsI
from LIM.training.run_state import RunState
from enum import Enum
from pathlib import Path
from datetime import datetime
import LIM.log as log
from LIM.models.modelI import Model
import LIM.log as log


def handle_OOM(func: Callable) -> bool:
    @functools.wraps(func)
    def inner(*args, **kwargs) -> Any:
        try:
            func(*args, **kwargs)
            return True
        except RuntimeError as e:
            log.warn(f"Cuda OOM: {e}")
            return False

    return inner


class BaseTrainer(ABC):
    """
    The general neural network training routine.
    For each network we train we need to make a new ___Trainer(BaseTrainer) that implements the abstract methods:
        - _custom_train_step(sample: Any) -> bool
        - _custom_val_step(sample: Any) -> bool
        - _custom_epoch_step() -> None
        - _custom_loss_log(mode: str) -> None
    """

    mode: Path
    device: torch.device
    state: RunState
    model: torch.nn.Module
    dataloader: torch.utils.data.DataLoader
    dataset: CloudDatasetsI
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler

    def __init__(self, model: torch.nn.Module, dataset: CloudDatasetsI, mode: Optional["Mode"] = None) -> None:
        BaseTrainer.device = torch.device(config.settings.DEVICE)
        self._load_model(model)
        self.BACKUP_DIR = mode.path(to=self.model.__class__.__name__)
        log.info(f"Running {mode._name_} training on {self.BACKUP_DIR}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.settings.TRAINER.LEARNING_RATE.VALUE)
        self.state = RunState(run_name=f"{self.BACKUP_DIR.parent.stem}/{self.BACKUP_DIR.stem}")
        self.__load_config(mode := mode if mode is not None else BaseTrainer.Mode.NEW)
        self.__load_dataloaders(dataset)

    def __load_config(self, mode: "Mode"):
        match mode:
            case BaseTrainer.Mode.NEW:
                config.settings.save(self.BACKUP_DIR)
            case BaseTrainer.Mode.FIXED | BaseTrainer.Mode.LATEST:
                log.info("Overwriting default settings")
                config.settings = config.Settings.load(self.BACKUP_DIR)
                self.model.load(run=self.BACKUP_DIR, suffix="latest")
                self.state.load(run=self.BACKUP_DIR, suffix="latest")
            case _:
                log.error(f"Invalid BaseTrainer.Mode given to BaseTrainer.__load_config: {mode}")
                raise ValueError

    def __load_dataloaders(self, dataset: CloudDatasetsI) -> None:
        self.train_set, self.val_set = (
            dataset.new_instance(CloudDatasetsI.SPLITS.TRAIN),
            dataset.new_instance(CloudDatasetsI.SPLITS.VAL),
        )

        self.train_loader = self.make_dataloader(
            self.train_set,
            num_workers=mp.cpu_count() - 4 if config.settings.TRAINER.MULTIPROCESSING else 0,
        )
        self.val_loader = self.make_dataloader(
            self.val_set,
            num_workers=2 if config.settings.TRAINER.MULTIPROCESSING else 0,
        )
        log.info(
            f"Loaded {dataset.__name__}: training set has {len(self.train_set)} samples, validation set has {len(self.val_set)} samples"
        )

    def __repr__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _load_model(self, model: Model) -> None: ...

    @abstractmethod
    def _custom_train_step(self, sample: Any) -> bool:
        """
        Runs a training step

        Args:
            sample: The sample returned by the dataloader
        Returns:
            bool: True if the step was executed succesfully
        """
        ...

    @abstractmethod
    def _custom_val_step(self, sample: Any) -> bool:
        """
        Runs a validation step

        Args:
            sample: The sample returned by the dataloader
        Returns:
            bool: True if the step was executed succesfully
        """
        ...

    @abstractmethod
    def _custom_epoch_step() -> None:
        """
        Anything you want to run after an epoch ends, e.g. a learning rate scheduler step
        """
        ...

    @abstractmethod
    def _custom_loss_log(self, mode: str) -> str:
        """
        Handles the logging of losses of the current step

        Args:
            mode: Either "train" or "val"
        """
        ...

    def train(self) -> None:
        for self.state.train.epoch in range(self.state.train.epoch, config.settings.TRAINER.EPOCHS):
            self.state.val.epoch = self.state.train.epoch
            self.optimizer.zero_grad()
            self.__clean_memory()
            self.__train_step()
            self.__val_step()
            self._custom_epoch_step()

    def __train_step(self) -> bool:
        self.model.train()
        with log.Live(
            log.Text.from_markup(self._custom_loss_log(mode="train")), refresh_per_second=2, console=log.console
        ) as train_live:
            for self.state.train.step, sample in enumerate(self.train_loader, start=self.state.train.step):
                self.__clean_memory()
                if not self._custom_train_step(sample):
                    self.train_set.force_downsample(sample)

                if self.state.on_accumulation_step or (
                    _ON_LAST_BATCH := (self.state.train.step + 1) == len(self.train_loader)
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.state.on_backup_step:
                    log.info(f"Making backup on step {self.state.train.step}")
                    self.model.save_async(run=self.BACKUP_DIR, suffix="latest")
                    self.state.save_async(run=self.BACKUP_DIR, suffix="latest")

                train_live.update(log.Text.from_markup(self._custom_loss_log(mode="train")))
                self.state.train.iteration += 1
        return True

    def __val_step(self) -> bool:
        self.model.eval()
        with torch.no_grad():
            with log.Live(
                log.Text.from_markup(self._custom_loss_log(mode="val")), refresh_per_second=2, console=log.console
            ) as val_live:
                for self.state.val.step, sample in enumerate(self.val_loader):
                    self.__clean_memory()
                    if not self._custom_val_step(sample=sample):
                        self.val_set.force_downsample(sample)

                    if self.state.val.on_best_iter:
                        log.info(f"Saving best model on step {self.state.train.step}")
                        self.model.save_async(run=self.BACKUP_DIR, suffix="best")
                        self.state.save_async(run=self.BACKUP_DIR, suffix="best")
                        self.state.val.on_best_iter = False

                    val_live.update(log.Text.from_markup(self._custom_loss_log(mode="val")))
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
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.settings.TRAINER.BATCH_SIZE,
            shuffle=True,
            collate_fn=partial(dataset.collate_fn),
            num_workers=num_workers,
            multiprocessing_context="spawn" if config.settings.TRAINER.MULTIPROCESSING else None,
            persistent_workers=True if config.settings.TRAINER.MULTIPROCESSING else False,
        )

    class Mode(Enum):
        """
        Handles the selection of the path where the backups will be created

        [WARN] Since Enums share a global state, this class (and therefore BaseTrainer) can't be used with parallel
        instances, because different instances will modify the same global value of _path.
        """

        NEW = "Create a new run"
        FIXED = "Choose a run with YYYYMMDD_HHMMSS format"
        LATEST = "Start with latest run present in the backups folder"

        def __init__(self, *args, **kwargs):
            self._path = None

        def dated(self, path: str) -> "BaseTrainer.Mode":
            self._path = path
            return self

        def path(self, to: str) -> Path:
            """
            This method references _model_name, which is only set in BaseTrainer.__init__, so this can't be
            called unless we
            """
            resulting_path = config.settings.TRAINER.BACKUP_DIR / to
            DATE_FORMAT = "%Y%m%d_%H%M%S"
            match self:
                case BaseTrainer.Mode.NEW:
                    if self._path is not None:
                        log.warn(f"Given date {self._path} will be ignored with selected mode NEW")

                    resulting_path = resulting_path / Path(datetime.now().strftime(DATE_FORMAT))
                case BaseTrainer.Mode.FIXED:
                    if not self._path is not None:
                        msg = "Mode fixed needs value to dir with the YYYYMMDD_HHMMSS format"
                        log.error("Mode fixed needs value to dir with the YYYYMMDD_HHMMSS format")
                        raise ValueError(msg)
                    try:
                        datetime.strptime(self._path, DATE_FORMAT)
                    except ValueError as e:
                        log.error(f"Gave mode FIXED a path with an invalid YYYYMMDD_HHMMSS format: {self._path}")
                        raise e
                    resulting_path = resulting_path / self._path
                case BaseTrainer.Mode.LATEST:
                    if self._path is not None:
                        log.warn(f"Given date {self._path} will be ignored with selected mode LATEST")

                    @staticmethod
                    def is_valid_timestamp(ts: str):
                        try:
                            datetime.strptime(ts, DATE_FORMAT)
                            return True
                        except ValueError:
                            return False

                    subdirs = [p for p in resulting_path.iterdir() if p.is_dir() and is_valid_timestamp(p.name)]

                    resulting_path = max(subdirs, key=lambda p: datetime.strptime(p.name, DATE_FORMAT))

            resulting_path.mkdir(parents=True, exist_ok=True)
            return resulting_path
