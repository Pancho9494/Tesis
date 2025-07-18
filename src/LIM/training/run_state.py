from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import aim
from config.config import settings
from pathlib import Path
import msgpack
import LIM.log as log
from LIM.training.threading import backup_executor
from os import PathLike


class RunState:
    @dataclass
    class Current:
        iteration: int = field(default=0)
        epoch: int = field(default=0)
        step: int = field(default=0)
        best_model: Dict[str, Any] = field(default_factory=dict)
        on_best_iter: bool = False

        @property
        def log_header(self) -> str:
            return (
                f"Epoch[[cyan]{self.epoch:02d}/{settings.TRAINER.EPOCHS:02d}[/cyan]]"
                + f"Step[[cyan]{self.step:02d}[/cyan]]"
                + f"Iter[[cyan]{self.iteration:02d}[/cyan]]"
            )

    run_name: str | Path | PathLike
    train: Current
    val: Current
    tracker: aim.Run

    def __init__(self, run_name: str, train: Current = Current(), val: Current = Current()):
        if settings.DISTRIBUTED.RANK == 0:
            self.tracker = aim.Run(experiment=run_name)
        self.train = train
        self.val = val

    @property
    def on_first_step(self) -> bool:
        return self.train.step == 0

    @property
    def on_accumulation_step(self) -> bool:
        return (self.train.step + 1) % settings.TRAINER.ACCUM_STEPS == 0

    @property
    def on_backup_step(self) -> bool:
        return self.train.step % settings.TRAINER.BACKUP_PERIOD == 0

    def load(self, run: str | Path | PathLike = "", suffix: str = "") -> None:
        """ """
        path = Path(f"{run}/run_state_{suffix}.msgpack")

        if not path.exists():
            log.error(f"Couldn't find RunState backup at {path}")
            return

        log.info(f"RunState loading backup from {path}")

        with path.open("rb") as f:
            data = msgpack.load(f, raw=False)

        self.train = RunState.Current(**data["train"])
        self.val = RunState.Current(**data["val"])
        if settings.DISTRIBUTED.RANK == 0:
            self.tracker = aim.Run(run_hash=data["tracker_hash"])

    def save(self, run: str = "", suffix: str = "") -> None:
        if settings.DISTRIBUTED.RANK != 0:
            return
        path = Path(f"{run}/run_state_{suffix}.msgpack")
        log.info(f"RunState saving backup to {path}")
        tracker_hash = self.tracker.hash if settings.DISTRIBUTED.RANK == 0 else None
        with path.open("wb") as f:
            data = msgpack.packb(
                {
                    "train": asdict(self.train),
                    "val": asdict(self.val),
                    "tracker_hash": tracker_hash,
                },
                use_bin_type=True,
            )
            f.write(data)

    def save_async(self, run: str | Path | PathLike = "", suffix: str = "") -> None:
        backup_executor.submit(self.save, run, suffix)
