import builtins

from rich import traceback, pretty, print
import torch
import config.config as config
import argparse


torch.multiprocessing.set_start_method("spawn", force=True)
traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def train_predator(pre_trained: bool = False) -> None:
    from LIM.training import BaseTrainer
    from LIM.models.PREDATOR import PREDATOR, PredatorTrainer
    from LIM.data.sets.threeDLoMatch import ThreeDLoMatch

    PredatorTrainer(
        mode=BaseTrainer.Mode.NEW,
        model=PREDATOR,
        dataset=ThreeDLoMatch,
    ).train()


def train_iae() -> None:
    from LIM.training import BaseTrainer
    from LIM.models.IAE import IAETrainer
    from LIM.models.PREDATOR import PREDATOR
    from LIM.data.sets.scanNet import ScanNet

    IAETrainer(
        mode=BaseTrainer.Mode.NEW,
        model=PREDATOR,
        dataset=ScanNet,
    ).train()


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
    match config.settings.MODEL.MODULE:
        case "IAE":
            train_iae()
        case "PREDATOR":
            train_predator()
        case _:
            raise ValueError(f"No model named {config.settings.MODEL.MODULE}")
