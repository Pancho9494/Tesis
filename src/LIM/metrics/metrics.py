from abc import ABC, abstractmethod
import aim
from dataclasses import dataclass, field
import torch
from typing import Any, Callable, Protocol, List

from config import settings


class TrainerStateProtocol(Protocol):
    class CurrentProtocol(Protocol):
        epoch: int
        step: int

    tracker: aim.Run
    current: CurrentProtocol


@dataclass
class Metric(ABC):
    """ """

    name: str
    context: str
    trainer_state: TrainerStateProtocol
    custom_function: Callable
    current: float = field(default=0.0)
    best: float = field(default=0.0)

    def track(self, new_value: float) -> None:
        self.current = new_value
        if self.current > self.best:
            self.best = self.current

    @property
    def on_best_iter(self) -> bool:
        return self.current == self.best

    def __call__(self, sample: Any) -> torch.Tensor:
        self.current = (loss := self.custom_function(sample)).item()
        if self.current > self.best:
            self.best = self.current
        self.trainer_state.tracker.track(
            self.current,
            name=self.name,
            step=self.trainer_state.current.step,
            epoch=self.trainer_state.current.epoch,
            context={"subset": self.context},
        )
        return loss


class Loss(ABC):
    """ """

    train: Metric
    val: Metric
    device = torch.device(settings.DEVICE)

    def __init__(self, trainer_state: TrainerStateProtocol) -> None:
        self.train = Metric(
            name=self.__class__.__name__,
            context="train",
            trainer_state=trainer_state,
            custom_function=self.__call__,
        )
        self.val = Metric(
            name=self.__class__.__name__,
            context="val",
            trainer_state=trainer_state,
            custom_function=self.__call__,
        )

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __call__(self, sample: Any) -> torch.Tensor: ...


class MultiLoss:
    train: Metric
    val: Metric
    losses: List[Loss]

    def __init__(self, losses: List[Loss]):
        self.losses = losses
        self.train = Metric(
            name="MutliLoss",
            context="train",
            trainer_state=losses[0].train.trainer_state,
            custom_function=lambda sample: sum(loss.train(sample) for loss in self.losses),
        )
        self.val = Metric(
            name="MutliLoss",
            context="val",
            trainer_state=losses[0].val.trainer_state,
            custom_function=lambda sample: sum(loss.val(sample) for loss in self.losses),
        )

    def __repr__(self) -> str:
        return f"MultiLoss({[loss for loss in self.losses]})"
