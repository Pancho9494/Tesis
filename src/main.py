import numpy as np
import builtins
from rich import traceback, pretty, print
import torch
import torch.utils.viz
from LIM.models.PREDATOR.predator import Predator
from LIM.data.sets.threeDLoMatch import ThreeDLoMatch
from LIM.models.trainer import PredatorTrainer

torch.multiprocessing.set_start_method("spawn", force=True)
traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def train_predator() -> None:
    PredatorTrainer(
        model=Predator(),
        dataset=ThreeDLoMatch(),
    ).train()


if __name__ == "__main__":
    train_predator()


# class Painter:
