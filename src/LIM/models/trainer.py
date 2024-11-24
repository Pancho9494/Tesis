import torch
import torch.nn.common_types
from torch.utils.data import DataLoader
import multiprocessing as mp
from LIM.data.structures.cloud import Cloud
from LIM.data.datasets.datasetI import CloudDatasetsI
from LIM.data.datasets.scanNet import collate_scannet
from LIM.data.structures.transforms import Downsample, BreakSymmetry, CenterZRandom, Noise
from LIM.metrics.losses import L1Loss, IOU
from aim import Run
from dataclasses import dataclass, field
import torchvision
from typing import Dict, List, Any, Union, Optional
from pathlib import Path
import copy
import gc
from functools import partial
from config import settings
import numpy as np


class Transforms:
    cloud_tf: torchvision.transforms.Compose
    implicit_tf: torchvision.transforms.Compose

    def __init__(self, cloud_tf: List[torch.nn.Module], implicit_tf: List[torch.nn.Module]) -> None:
        self.cloud_tf = torchvision.transforms.Compose(cloud_tf)
        self.implicit_tf = torchvision.transforms.Compose(implicit_tf)


class Trainer:
    __device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @dataclass
    class Current:
        epoch: int = field(default=0)
        step: int = field(default=0)
        best_model: Dict[str, Any] = field(default_factory=dict)
        train_loss: float = field(default=np.inf)
        val_loss: float = field(default=0.0)
        best_val_loss: float = field(default=0.0)

    run: Run
    current: Current
    model: torch.nn.Module
    dataloader: DataLoader
    dataset: CloudDatasetsI
    optimizer: torch.optim.Optimizer
    transforms: Dict[str, Transforms]

    def __init__(self, tag: str, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        self.tag = tag
        self.model = model.to(self.__device)
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=settings.TRAINER.LEARNING_RATE,
        )
        self.__make_split()
        self.current = Trainer.Current()
        self.run = Run(experiment="IAE Training")
        self.run["trainer"] = settings.TRAINER.__dict__
        # self.run["model"] = settings.MODEL.__dict__

    def train(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.multiprocessing.set_start_method("spawn", force=True)
        cloud: Cloud
        implicit: Cloud
        for self.current.epoch in range(self.current.epoch, settings.TRAINER.EPOCHS):
            for self.current.step, (cloud, implicit) in enumerate(self.train_loader, self.current.step):
                self.__train_step(cloud, implicit)
                self.__val_step()
                self.__print()
                self.__backup_model(self.model.state_dict(), "latest")

    def load_model(self, path: Union[str, Path]) -> None:
        print(f"Loading model from {path}")
        backup: Dict[str, Any] = torch.load(path, weights_only=False)
        self.tag = backup["tag"]
        self.model.load_state_dict(backup["model_state_dict"])
        self.optimizer.load_state_dict(backup["optimizer_state_dict"])
        self.current.step = backup["current_step"]
        self.current.epoch = backup["current_epoch"]
        self.current.train_loss = backup["current_train_loss"]
        self.current.val_loss = backup["current_val_loss"]

    def __backup_model(self, model: Dict[str, Any], name: str) -> None:
        if self.current.step == 0 or self.current.step % settings.TRAINER.BACKUP_PERIOD != 0:
            return
        torch.save(
            {
                "tag": self.tag,
                "model_state_dict": model,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "current_step": self.current.step,
                "current_epoch": self.current.epoch,
                "current_train_loss": self.current.train_loss,
                "current_val_loss": self.current.val_loss,
                "params": settings.__dict__,
            },
            f"./weights/{self.tag}_{name}.tar",
        )

    def __train_step(self, cloud: Cloud, implicit: Cloud) -> None:
        self.model.train()
        predicted_df = self.model(cloud, implicit)
        step_loss = self.train_loss(predicted_df, implicit.features)
        accum_loss = step_loss / settings.TRAINER.ACCUM_STEPS
        accum_loss.backward()
        self.run.track(
            step_loss.item(),
            name="Training Loss",
            step=self.current.epoch + self.current.step + 1,
            context={"subset": "train"},
        )
        accumulation_done = (self.current.step + 1) % settings.TRAINER.ACCUM_STEPS == 0
        on_last_batch = (self.current.step + 1) == len(self.train_loader)
        if accumulation_done or on_last_batch:
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.current.train_loss = step_loss.item()

    def __val_step(self) -> None:
        if self.current.step == 0 or self.current.step % settings.TRAINER.VALIDATION_PERIOD != 0:
            return

        self.model.eval()
        with torch.no_grad():
            cloud, implicit = next(self.val_loader)
            predicted_df = self.model(cloud, implicit)
            loss = self.val_loss(predicted_df, implicit.features)
            self.run.track(
                loss.item(),
                name="Validation IOU",
                step=self.current.epoch + self.current.step + 1,
                context={"subset": "val"},
            )
            self.current.val_loss = loss.item()
            if self.current.val_loss > self.current.best_val_loss:
                self.current.best_val_loss = self.current.val_loss
                print(f"Backing up best model [{self.current.val_loss}]")
                self.current.best_model = self.model.state_dict()
                self.__backup_model(self.current.best_model, "best")

    def __make_split(self) -> None:
        train_set, val_set = torch.utils.data.random_split(
            self.dataset, [1 - settings.TRAINER.VALIDATION_SPLIT, settings.TRAINER.VALIDATION_SPLIT]
        )
        train_set.dataset, val_set.dataset = copy.deepcopy(self.dataset), copy.deepcopy(self.dataset)

        print(f"Train split has {len(train_set.indices)} point clouds")
        print(f"Validation split has {len(val_set.indices)} point clouds")
        self.train_loss = L1Loss(reduction="none")
        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=settings.TRAINER.BATCH_SIZE,
            shuffle=True,
            collate_fn=partial(
                collate_scannet,
                cloud_tf=[
                    CenterZRandom(base_ratio=0.5),
                    Downsample(n_points=4096),
                    Noise(noise=0.005),
                ],
                implicit_tf=[
                    BreakSymmetry(std_dev=1e-4),
                    Downsample(n_points=2048),
                ],
            ),
            num_workers=mp.cpu_count() - 4 if settings.TRAINER.MULTIPROCESSING else 0,
            multiprocessing_context="spawn" if settings.TRAINER.MULTIPROCESSING else None,
            pin_memory=True,
        )
        self.val_loss = IOU(threshold=0.5)
        self.val_loader = iter(
            DataLoader(
                dataset=val_set,
                batch_size=1,
                shuffle=False,
                collate_fn=partial(
                    collate_scannet,
                    cloud_tf=[
                        CenterZRandom(base_ratio=0.5),
                        Downsample(n_points=4096),
                        Noise(noise=0.005),
                    ],
                    implicit_tf=[
                        BreakSymmetry(std_dev=10e-4),
                    ],
                ),
                num_workers=2 if settings.TRAINER.MULTIPROCESSING else 0,
                multiprocessing_context="spawn" if settings.TRAINER.MULTIPROCESSING else None,
                pin_memory=True,
            )
        )

    def __print(self) -> None:
        step_id = f"Epoch[{(self.current.epoch):02d}]\tIter[{(self.current.step):02d}]"
        step_id += f"\tTrain Loss[{self.current.train_loss:5.2f}]"
        step_id += (
            f"\tVal Loss[{self.current.val_loss:5.2f}]"
            if self.current.step != 0 and self.current.step % settings.TRAINER.VALIDATION_PERIOD == 0
            else ""
        )
        print(step_id)
