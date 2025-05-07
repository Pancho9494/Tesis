from rich.console import Console
from rich.live import Live
from rich.text import Text

console = Console()


def info(msg: str) -> None:
    console.log(f"[bold green][INFO][/bold green] {msg}", _stack_offset=2)


def warn(msg: str) -> None:
    console.log(f"[bold yellow][WARN][/bold yellow] {msg}", _stack_offset=2)


def error(msg: str) -> None:
    console.log(f"[bold red][ERROR][/bold red] {msg}", _stack_offset=2)
