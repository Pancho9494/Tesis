import torch
import numpy as np
import random

import torch.utils.viz
from LIM.models.PREDATOR.predator import Predator
from LIM.data.sets.threeDLoMatch import ThreeDLoMatch
from LIM.models.trainer import PredatorTrainer
import builtins
from rich import traceback, pretty, print
from config import settings

torch.multiprocessing.set_start_method("spawn", force=True)
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def train_predator() -> None:
    # torch.cuda.memory._record_memory_history()
    dataset = ThreeDLoMatch()
    model = Predator().to(settings.DEVICE)

    trainer = PredatorTrainer("PREDATOR_Training", model, dataset)
    trainer.train()
    # torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    train_predator()
