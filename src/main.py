import builtins
from rich import traceback, pretty, print
import torch
import config.config as config

torch.multiprocessing.set_start_method("spawn", force=True)
traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def train_predator() -> None:
    config.settings = config.Settings.from_yaml("./src/config/LIM.yaml")

    from LIM.training import BaseTrainer
    from LIM.models.PREDATOR import PREDATOR, PredatorTrainer
    from LIM.data.sets.threeDLoMatch import ThreeDLoMatch

    PredatorTrainer(
        mode=BaseTrainer.Mode.NEW,
        model=PREDATOR,
        dataset=ThreeDLoMatch,
    ).train()


def train_iae() -> None:
    config.settings = config.Settings.from_yaml("./src/config/IAE.yaml")

    from LIM.training import BaseTrainer
    from LIM.models.IAE import IAETrainer
    from LIM.models.PREDATOR import PREDATOR
    from LIM.data.sets.scanNet import ScanNet

    IAETrainer(
        mode=BaseTrainer.Mode.NEW,
        model=PREDATOR,
        dataset=ScanNet,
    ).train()


async def clean_scannet() -> None:
    from LIM.data.sets.scanNet import ScanNet

    await ScanNet.clean_bad_files()


if __name__ == "__main__":
    # train_predator()
    train_iae()
    # asyncio.run(clean_scannet())

    from LIM.training.threading import backup_executor

    backup_executor.shutdown(wait=True)
