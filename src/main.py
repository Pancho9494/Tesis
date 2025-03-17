import builtins
from rich import traceback, pretty, print
import torch
import torch.utils.viz

# from LIM.models.IAE import IAE, IAETrainer
from LIM.models.PREDATOR import PREDATOR, PredatorTrainer, Encoder

# from LIM.data.sets.scanNet import ScanNet
from LIM.data.sets.threeDLoMatch import ThreeDLoMatch

torch.multiprocessing.set_start_method("spawn", force=True)
traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def train_predator() -> None:
    PredatorTrainer(
        model=PREDATOR(),
        dataset=ThreeDLoMatch(),
    ).train()


# def train_iae() -> None:
#     IAETrainer(
#         model=IAE(encoder=Encoder),
#         dataset=ScanNet(),
#     ).train()


if __name__ == "__main__":
    train_predator()
    # train_iae()
