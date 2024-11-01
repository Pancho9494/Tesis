import torch
from torch.utils.data import DataLoader
import multiprocessing as mp
from LIM.data.structures.cloud import Cloud
from LIM.data.datasets.datasetI import CloudDatasetsI
from aim import Run
from dataclasses import dataclass, field
import open3d as o3d


class Trainer:
    @dataclass
    class Params:
        BATCH_SIZE: int = field(default=1)
        LEARNING_RATE: float = field(default=1e-4)
        EPOCHS: int = field(default=100)

    params: Params
    model: torch.nn.Module
    dataset: CloudDatasetsI
    dataloader: DataLoader
    optimizer: torch.optim.Optimizer
    run: Run

    def __init__(self, model: torch.nn.Module, dataset: CloudDatasetsI) -> None:
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = Trainer.Params(BATCH_SIZE=1, LEARNING_RATE=1e-4, EPOCHS=100)
        self.model = model.to(self.__device)
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.params.LEARNING_RATE)
        self.loss = torch.nn.L1Loss(reduction="none")

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.params.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset.collate,
            # num_workers=mp.cpu_count() - 2,
            # multiprocessing_context="spawn",
        )

        # self.run = Run(experiment="IAE Training")
        # self.run["hyperparameters"] = self.params.__dict__

    def train(self, plot: bool = False) -> None:
        for epoch in range(self.params.EPOCHS):
            self.model.train()
            avg_loss = self.__train_step(epoch)
            self.__val_step(epoch)

    def __train_step(self, epoch_idx: int) -> float:
        cloud: Cloud
        implicit: Cloud
        running_loss: float = 0.0
        last_loss: float = 0.0

        return 0.0
        # for it, (cloud, implicit) in enumerate(self.dataloader):
        #     step_id = f"Epoch[{(epoch_idx + 1):02d}]\tIter[{(it + 1):02d}]"
        #     self.optimizer.zero_grad()

        #     # try:
        #     predicted_df = self.model(cloud, implicit)
        #     # except RuntimeError as e:
        #     #     self.run.log_error(f"{step_id} Runtime error: {e}")
        #     #     continue

        #     loss = self.loss(predicted_df, torch.tensor(implicit.features).unsqueeze(0)).sum(-1).mean()
        #     loss.backward()
        #     self.run.log_info(f"{step_id}\tInstant Loss[{loss.item():5.2f}]")

        #     print(f"{step_id}\tInstant Loss[{loss.item():5.2f}]")

        #     self.optimizer.step()

        #     running_loss += loss.item()
        #     if it % 1000 == 999:
        #         last_loss = running_loss / 1000
        #         self.run.log_info(f"{step_id}\tAvg Loss[{last_loss:5.2f}]")

        #         self.run.track(
        #             last_loss,
        #             name="L1Loss",
        #             step=epoch_idx * len(self.dataloader) + it + 1,
        #             context={"subset": "train"},
        #         )
        #         running_loss = 0

        # return last_loss

    def __val_step(self, epoch_idx: int) -> None:
        self.model.eval()
        with torch.no_grad():
            ...  # run inference on validation loader here
