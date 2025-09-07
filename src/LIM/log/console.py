from rich.console import Console
from rich.live import Live
from rich.text import Text


class Logger:
    console: Console
    _run_hash: str | None

    def __init__(self) -> None:
        self.console = Console()
        self._run_hash = None

    @property
    def Live(self) -> type[Live]:
        return Live

    @property
    def Text(self) -> type[Text]:
        return Text

    @property
    def run_hash(self) -> str | None:
        return self._run_hash

    @run_hash.setter
    def run_hash(self, value: str) -> None:
        self._run_hash = value

    def _format_msg(self, msg: str) -> str:
        if self._run_hash is not None:
            return f"[bold blue][{self._run_hash}][/bold blue] {msg}"
        return msg

    def info(self, msg: str) -> None:
        self.console.log(self._format_msg(f"[bold green][INFO][/bold green] {msg}"), _stack_offset=2)

    def warn(self, msg: str) -> None:
        self.console.log(self._format_msg(f"[bold yellow][WARN][/bold yellow] {msg}"), _stack_offset=2)

    def error(self, msg: str) -> None:
        self.console.log(self._format_msg(f"[bold red][ERROR][/bold red] {msg}"), _stack_offset=2)


log = Logger()
