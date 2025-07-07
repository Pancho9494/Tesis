import argparse
import builtins
import atexit
import torch
from rich import pretty, print, traceback

import config.config as config

torch.multiprocessing.set_start_method("spawn", force=True)
traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def train_predator(mode: "BaseTrainer.Mode", pre_trained: bool = False) -> None:
    from LIM.data.sets.threeDLoMatch import ThreeDLoMatch
    from LIM.models.PREDATOR import PREDATOR, PredatorTrainer

    trainer = PredatorTrainer(
        mode=mode,
        model=PREDATOR,
        dataset=ThreeDLoMatch,
    )
    atexit.register(trainer.cleanup)
    trainer.train()


def train_iae(mode: "BaseTrainer.Mode") -> None:
    from LIM.data.sets.scanNet import ScanNet
    from LIM.models.IAE import IAETrainer
    from LIM.models.PREDATOR import PREDATOR

    trainer = IAETrainer(
        mode=mode,
        model=PREDATOR,
        dataset=ScanNet,
    )
    atexit.register(trainer.cleanup)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Trainer",
        description="Launch ",
        add_help=True,
    )
    parser.add_argument(
        "config_file",
        help="Path to the configuration file",
        default="foo",
    )
    args = parser.parse_args()
    config.settings = config.Settings.from_yaml(args.config_file)
    from LIM.training import BaseTrainer

    match config.settings.TRAINER.MODE:
        case config.AvailableTrainingModes.NEW:
            mode = BaseTrainer.Mode.NEW
        case config.AvailableTrainingModes.LATEST:
            mode = BaseTrainer.Mode.LATEST
    match config.settings.MODEL.MODULE:
        case "IAE":
            train_iae(mode)
        case "PREDATOR":
            train_predator(mode)
        case _:
            raise ValueError(f"No model named {config.settings.MODEL.MODULE}")
