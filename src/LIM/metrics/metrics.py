from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol
import copy
import aim
import msgpack
import torch

import LIM.log as log
from config.config import settings


class TrainerStateProtocol(Protocol):
    class CurrentProtocol(Protocol):
        iteration: int
        epoch: int
        step: int

    tracker: aim.Run
    train: CurrentProtocol
    val: CurrentProtocol


@dataclass
class Metric(ABC):
    """ """

    name: str
    subset: str
    context: Dict[str, str]
    trainer_state: TrainerStateProtocol
    custom_function: Callable

    also_track: List[str] = field(default_factory=list)
    best: float = field(default=0.0)
    current: float = field(default=0.0)
    total_sum: float = field(default=0.0)
    average: float = field(default=0.0)

    def __name__(self) -> str:
        return self.name

    @property
    def on_best_iter(self) -> bool:
        return self.current >= self.best

    def __call__(self, sample: Any) -> torch.Tensor:
        self.current = (loss := self.custom_function(sample)).item()
        if self.current > self.best:
            self.best = self.current

        counters = getattr(self.trainer_state, self.subset)
        N = counters.iteration + 1
        self.total_sum += self.current
        self.average = self.total_sum / N
        if settings.DISTRIBUTED.RANK == 0:
            self.trainer_state.tracker.track(
                self.current,
                name=self.name,
                step=counters.iteration,
                epoch=counters.epoch,
                context=self.context | {"track": "current"},
            )

            for value in self.also_track:
                self.trainer_state.tracker.track(
                    getattr(self, value),
                    name=self.name,
                    step=counters.iteration,
                    epoch=counters.epoch,
                    context=self.context | {"track": value},
                )

        return loss

    def __repr__(self) -> str:
        return f"{self.name} [[cyan]{self.current:5.4f}[/cyan]]"

    def get(self, value: str) -> float:
        assert (value := value.lower().strip()) in ["best", "current", "total_sum", "average"]
        return getattr(self, value)

    def to_dict(self) -> dict[str, Any]:
        data = copy.copy(self.__dict__)
        del data["trainer_state"]
        del data["custom_function"]
        return data

    def from_dict(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Loss(ABC):
    """ """

    train: Metric
    val: Metric
    device: torch.device

    def __init__(self, trainer_state: TrainerStateProtocol, also_track: List[str] = [], y0to1: bool = False) -> None:
        self.device = torch.device(settings.DEVICE)
        self.train = Metric(
            name=self.__class__.__name__,
            subset="train",
            context={
                "subset": "train",
                "y0to1": y0to1,
            },
            trainer_state=trainer_state,
            custom_function=self.__call__,
            also_track=also_track,
        )
        self.val = Metric(
            name=self.__class__.__name__,
            subset="val",
            context={
                "subset": "val",
                "y0to1": y0to1,
            },
            trainer_state=trainer_state,
            custom_function=self.__call__,
            also_track=also_track,
        )

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __call__(self, sample: Any) -> torch.Tensor: ...

    def save(self, run: str = "", suffix: str = "") -> None:
        if settings.DISTRIBUTED.RANK != 0:
            return
        path = Path(f"{run}/{self.__class__.__name__}_{suffix}.msgpack")
        log.info(f"{self.__class__.__name__} saving backup to {path}")
        with path.open("wb") as f:
            data = msgpack.packb(
                {
                    "train": self.train.to_dict(),
                    "val": self.val.to_dict(),
                },
                use_bin_type=True,
            )
            f.write(data)

    def load(self, run: str | Path = "", suffix: str = "") -> None:
        path = Path(f"{run}/{self.__class__.__name__}_{suffix}.msgpack")
        if not path.exists():
            log.error(f"Couldn't find {self.__class__.__name__} backup at {path}")
            return
        log.info(f"{self.__class__.__name__} loading backup from {path}")

        with path.open("rb") as f:
            data = msgpack.load(f, raw=False)

        self.train.from_dict(data["train"])
        self.val.from_dict(data["val"])


class MultiLoss:
    _train: Metric
    _val: Metric
    losses: List[Loss]

    def __init__(self, losses: List[Loss]):
        self.losses = losses
        self._train = Metric(
            name="MultiLoss",
            subset="train",
            context={
                "subset": "train",
                "y0to1": False,
            },
            trainer_state=losses[0].train.trainer_state,
            custom_function=lambda sample: sum(loss.train(sample) for loss in self.losses),
            also_track=["average"],
        )
        self._val = Metric(
            name="MultiLoss",
            subset="val",
            context={
                "subset": "val",
                "y0to1": False,
            },
            trainer_state=losses[0].val.trainer_state,
            custom_function=lambda sample: sum(loss.val(sample) for loss in self.losses),
            also_track=["average"],
        )

    def __repr__(self) -> str:
        return f"MultiLoss({[loss for loss in self.losses]})"

    @property
    def train(self) -> Metric:
        return self._train

    @property
    def val(self) -> Metric:
        return self._val
