import torch

# import torch.nn.common_types
from torch.utils.data import DataLoader
import multiprocessing as mp
from LIM.data.structures.cloud import Cloud
from LIM.data.datasets.datasetI import CloudDatasetsI
from LIM.data.datasets.scanNet import collate_scannet
from LIM.data.structures.transforms import Downsample, BreakSymmetry, CenterZRandom, Noise
from LIM.metrics.losses import Loss, L1Loss, IOU
from aim import Run
from dataclasses import dataclass, field
import torchvision
from typing import Dict, List, Any, Union
from pathlib import Path
import copy
import gc
from functools import partial
from config import settings
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


class Transforms:
    cloud_tf: torchvision.transforms.Compose
    implicit_tf: torchvision.transforms.Compose

    def __init__(self, cloud_tf: List[torch.nn.Module], implicit_tf: List[torch.nn.Module]) -> None:
        self.cloud_tf = torchvision.transforms.Compose(cloud_tf)
        self.implicit_tf = torchvision.transforms.Compose(implicit_tf)


class Trainer:
    __device: torch.device = torch.device(settings.DEVICE)

    @dataclass
    class Current:
        epoch: int = field(default=0)
        step: int = field(default=0)
        best_model: Dict[str, Any] = field(default_factory=dict)

    run: Run
    current: Current
    model: torch.nn.Module
    dataloader: DataLoader
    dataset: CloudDatasetsI
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
    transforms: Dict[str, Transforms]
    l1_loss: Loss
    iou_loss: Loss

    def __init__(self, tag: str, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        self.tag = tag
        self.model = model.to(self.__device)
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=settings.TRAINER.LEARNING_RATE,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=settings.TRAINER.EPOCHS,
            eta_min=1e-4,
        )
        self.l1_loss = L1Loss(reduction="none", accum_steps=settings.TRAINER.ACCUM_STEPS)
        self.iou_loss = IOU(threshold=0.5)

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
                self.__log(show=True)
                self.__backup_model(self.model.state_dict(), "latest")
            self.scheduler.step()
            print(f"Current learning rate: {self.scheduler.get_last_lr()}")

    def load_model(self, path: Union[str, Path]) -> None:
        print(f"Loading model from {path}")
        backup: Dict[str, Any] = torch.load(path, weights_only=False)
        self.tag = backup["tag"]
        self.model.load_state_dict(backup["model_state_dict"])
        self.optimizer.load_state_dict(backup["optimizer_state_dict"])
        self.current.step = backup["current_step"]
        self.current.epoch = backup["current_epoch"]

    def __backup_model(self, model: Dict[str, Any], name: str) -> None:
        on_first_step: bool = self.current.step == 0
        on_backup_step: bool = self.current.step % settings.TRAINER.BACKUP_PERIOD == 0
        if on_first_step or not on_backup_step:
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

    def __train_step(self, cloud: Cloud, implicit: Cloud) -> None:
        self.model.train()
        predicted_df = self.model(cloud, implicit)
        self.l1_loss.train(predicted_df, implicit.features, with_grad=True)
        self.iou_loss.train(predicted_df, implicit.features)

        accumulation_done = (self.current.step + 1) % settings.TRAINER.ACCUM_STEPS == 0
        on_last_batch = (self.current.step + 1) == len(self.train_loader)
        if accumulation_done or on_last_batch:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def __val_step(self) -> None:
        on_first_step: bool = self.current.step == 0
        on_validation_step: bool = self.current.step % settings.TRAINER.VALIDATION_PERIOD == 0
        if on_first_step or not on_validation_step:
            return

        self.model.eval()
        with torch.no_grad():
            cloud, implicit = next(self.val_loader)
            predicted_df = self.model(cloud, implicit)
            best_l1_loss: bool = self.l1_loss.val(predicted_df, implicit.features)
            best_iou: bool = self.iou_loss.val(predicted_df, implicit.features)

            if best_iou:
                print(f"Backing up best model [{self.iou_loss.get('val')}]")
                self.current.best_model = self.model.state_dict()
                self.__backup_model(self.current.best_model, "best")

    def __make_split(self) -> None:
        train_set, val_set = torch.utils.data.random_split(
            self.dataset, [1 - settings.TRAINER.VALIDATION_SPLIT, settings.TRAINER.VALIDATION_SPLIT]
        )
        train_set.dataset, val_set.dataset = copy.deepcopy(self.dataset), copy.deepcopy(self.dataset)

        print(f"Train split has {len(train_set.indices)} point clouds")
        print(f"Validation split has {len(val_set.indices)} point clouds")
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

    def __log(self, show: bool = False) -> None:
        for loss in [self.l1_loss, self.iou_loss]:
            for mode in ["Training", "Validation"]:
                self.run.track(
                    loss.get(mode),
                    name=str(loss),
                    step=self.current.epoch + self.current.step + 1,
                    context={
                        "subset": mode.lower(),
                    },
                )

        if not show:
            return

        step_id = f"Epoch[{(self.current.epoch):02d}]\tIter[{(self.current.step):02d}]"
        step_id += f"\tL1LossTrain[{self.l1_loss.get('train'):5.2f}] IOUTrain[{self.iou_loss.get('train'):5.2f}]"
        step_id += (
            f"\tL1LossVal[{self.l1_loss.get('val'):5.2f}] IOUVal[{self.iou_loss.get('val'):5.2f}]"
            if self.current.step != 0 and self.current.step % settings.TRAINER.VALIDATION_PERIOD == 0
            else ""
        )
        print(step_id)


# Maybe the loss objects should keep track, separately, of their validation period
