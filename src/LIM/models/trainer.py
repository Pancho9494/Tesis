import torch
import torch.nn.common_types
from torch.utils.data import DataLoader
import multiprocessing as mp
from LIM.data.structures.cloud import Cloud
from LIM.data.datasets.datasetI import CloudDatasetsI
from aim import Run
from dataclasses import dataclass, field
from LIM.data.structures.transforms import Downsample, BreakSymmetry, CenterZRandom, Noise
import torchvision
from typing import Dict, List


class Transforms:
    cloud_tf: torchvision.transforms.Compose
    implicit_tf: torchvision.transforms.Compose

    def __init__(self, cloud_tf: List[torch.nn.Module], implicit_tf: List[torch.nn.Module]) -> None:
        self.cloud_tf = torchvision.transforms.Compose(cloud_tf)
        self.implicit_tf = torchvision.transforms.Compose(implicit_tf)


class Trainer:
    # __device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    __device: torch.device = torch.device("cpu")

    @dataclass
    class Params:
        BATCH_SIZE: int = field(default=1)
        LEARNING_RATE: float = field(default=1e-4)
        EPOCHS: int = field(default=100)
        ACCUM_STEPS: int = field(default=16)
        VALIDATION_PERIOD: int = field(default=10000)

    params: Params
    model: torch.nn.Module
    dataset: CloudDatasetsI
    dataloader: DataLoader
    optimizer: torch.optim.Optimizer
    run: Run
    current_epoch: int
    current_step: int
    transforms: Dict[str, Transforms]

    def __init__(self, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        self.params = Trainer.Params(
            BATCH_SIZE=16,
            LEARNING_RATE=1e-4,
            EPOCHS=100,
            ACCUM_STEPS=1,
        )
        self.model = model.to(self.__device)
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.params.LEARNING_RATE)
        self.loss = torch.nn.L1Loss(reduction="none")
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.params.BATCH_SIZE,
            shuffle=True,
            collate_fn=dataset.collate,
            num_workers=mp.cpu_count() - 2,
            multiprocessing_context="spawn",
        )

        self.transforms = {
            "train": Transforms(
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
            "val": Transforms(
                cloud_tf=[
                    CenterZRandom(base_ratio=0.5),
                    Downsample(n_points=4096),
                    Noise(noise=0.005),
                ],
                implicit_tf=[
                    BreakSymmetry(std_dev=10e-4),
                    # points_file gets subsample
                    # points_iou_file gets nothing
                    # poitns iou is called in validation
                ],
            ),
        }

        self.run = Run(experiment="IAE Training")
        self.run["trainer"] = self.params.__dict__
        self.run["model"] = self.model.params.__dict__

    def train(self) -> None:
        cloud: Cloud
        implicit: Cloud
        for epoch in range(self.params.EPOCHS):
            self.current_epoch = epoch
            for step, (cloud, implicit) in enumerate(self.dataloader):
                self.current_step = step
                self.__train_step(cloud, implicit)

                if step % self.params.VALIDATION_PERIOD == 0:
                    self.__val_step(cloud, implicit)

    def __train_step(self, cloud: Cloud, implicit: Cloud) -> None:
        self.model.train()
        step_id = f"Epoch[{(self.current_epoch + 1):02d}]\tIter[{(self.current_step + 1):02d}]"
        cloud, implicit = self.transforms["train"].cloud_tf(cloud), self.transforms["train"].implicit_tf(implicit)

        predicted_df = self.model(cloud, implicit)
        loss = self.loss(predicted_df, implicit.features).sum(-1).mean() / self.params.ACCUM_STEPS
        loss.backward()
        self.run.track(
            loss.item(),
            name="Instant Training L1Loss",
            step=self.current_epoch * len(self.dataloader) + self.current_step + 1,
            context={"subset": "train"},
        )
        accumulation_done = (self.current_step + 1) % self.params.ACCUM_STEPS == 0
        on_last_batch = (self.current_step + 1) == len(self.dataloader)
        if accumulation_done or on_last_batch:
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.run.log_info(f"{step_id}\tTrain L1Loss[{loss.item():5.2f}]")
            self.run.track(
                loss.item(),
                name="Train L1Loss",
                step=self.current_epoch * len(self.dataloader) + self.current_step + 1,
                context={"subset": "train"},
            )

    def __val_step(self, cloud: Cloud, implicit: Cloud) -> None:
        self.model.eval()
        with torch.no_grad():
            cloud, implicit = self.transforms["val"].cloud_tf(cloud), self.transforms["val"].implicit_tf(implicit)
            predicted_df = self.model(
                cloud, implicit
            )  # here implicit should come from the points_iou file, which apparently is the same one in teh case of
            # dgcnn_semseg?, only difference here would be the transformations applied to it

            # TODO: add iou loss
