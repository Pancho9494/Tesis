import builtins
from rich import traceback, pretty, print
from LIM.data.sets.bunny import Bunny

traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def main():
    dataset = Bunny()


if __name__ == "__main__":
    main()
