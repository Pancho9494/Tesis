import argparse
import builtins

from rich import pretty, print, traceback

import config.config as config
from LIM.log import log

traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def main():
    from LIM.data.sets.threeDLoMatch import ThreeDLoMatch
    from LIM.models.evaluator import Evaluator
    from LIM.models.PREDATOR import PREDATOR

    evaluate = Evaluator(
        model=PREDATOR(),
        dataset=ThreeDLoMatch.new_instance(split=ThreeDLoMatch.SPLITS.TEST),
    )
    evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Evaluator",
        description="Test model performance",
        add_help=True,
    )
    parser.add_argument(
        "config_file",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    config.settings = config.Settings.from_yaml(args.config_file)
    log.info(f"Loaded config file from {args.config_file}\n{config.settings=}")
    main()
