from __future__ import annotations

import argparse
import atexit
import builtins

import torch
from rich import pretty, print, traceback

import config.config as config

torch.multiprocessing.set_start_method("spawn", force=True)
traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def train_predator(mode: BaseTrainer.Mode, pre_trained: bool = False) -> None:
    from LIM.data.sets.threeDLoMatch import ThreeDLoMatch
    from LIM.models.PREDATOR import PREDATOR, PredatorTrainer

    # ThreeDLoMatch.make_toy_pkl() # I removed `self.dir` from the __parse_info method
    trainer = PredatorTrainer(
        mode=mode,
        model=PREDATOR,
        dataset=ThreeDLoMatch,
    )
    atexit.register(trainer.cleanup)
    trainer.train()


def train_iae(mode: BaseTrainer.Mode) -> None:
    from LIM.data.sets.scanNet import ScanNet
    from LIM.models.IAE import IAETrainer
    from LIM.models.PREDATOR import PREDATOR

    # ScanNet.make_toy_lst()
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

    if config.settings.TRAINER.MODE == config.AvailableTrainingModes.NEW:
        mode = BaseTrainer.Mode.NEW
    elif config.settings.TRAINER.MODE == config.AvailableTrainingModes.LATEST:
        mode = BaseTrainer.Mode.LATEST
    else:
        raise ValueError(f"Selected unexpected mode: {config.settings.TRAINER.MODE=}")

    if "IAE" in config.settings.MODEL.MODULE:
        train_iae(mode)
    elif "PREDATOR" in config.settings.MODEL.MODULE:
        train_predator(mode)
    else:
        raise ValueError(f"No model named {config.settings.MODEL.MODULE}")
