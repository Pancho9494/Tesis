import builtins
from rich import traceback, pretty, print
from LIM.models.PREDATOR import PREDATOR
from LIM.data.sets.threeDLoMatch import ThreeDLoMatch
from LIM.models.evaluator import Evaluator

traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def main():
    evaluate = Evaluator(
        model=PREDATOR(),
        dataset=ThreeDLoMatch.new_instance(split=ThreeDLoMatch.SPLITS.TEST),
    )
    evaluate()


if __name__ == "__main__":
    main()
