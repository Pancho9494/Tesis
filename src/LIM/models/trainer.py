import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import multiprocessing as mp
from LIM.data.structures.cloud import Cloud
from typing import List
from LIM.data.datasets.datasetI import CloudDatasetsI


class Plotter:
    def __init__(self, activate: bool) -> None:
        self.activate = activate
        self.figure, self.ax = plt.subplots()
        self.epochs = []
        self.train_losses = []
        self.validation_losses = []

        plt.ion()

    def draw(self) -> None:
        if not self.activate:
            return
        plt.plot(self.epochs, self.train_losses)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.show()


class Trainer:
    model: torch.nn.Module
    dataset: CloudDatasetsI
    dataloader: DataLoader
    optimizer: torch.optim.Optimizer
    plotter: Plotter
    logger: SummaryWriter

    def __init__(self, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        self.model = model
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.loss = torch.nn.L1Loss(reduction="none")
        self.logger = SummaryWriter("./logs")

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=mp.cpu_count() - 2,
            collate_fn=dataset.collate,
        )

    def train(self, plot: bool = False) -> None:
        cloud: Cloud
        implicit: Cloud

        self.model.train()
        for it, (cloud, implicit) in enumerate(self.dataloader):
            predicted_df = self.model(cloud, implicit)
            loss = self.loss(predicted_df, torch.tensor(implicit.features).unsqueeze(0)).sum(-1).mean()
            loss.backward()
            self.optimizer.step()

            print(f"[Epoch {it:02d}]\tL1Loss({loss.item():5.2f})")

            self.logger.add_scalar("train/loss", loss, it)
