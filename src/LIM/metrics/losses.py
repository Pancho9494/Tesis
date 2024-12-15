import torch
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class Metric:
    """
    Class that tracks the best value for a loss
    """

    tensor: torch.Tensor
    current: float
    best: float

    def set_tensor(self, new_tensor: torch.Tensor) -> bool:
        self.current = new_tensor.item()
        if self.current > self.best:
            self.best = self.current
            return True
        return False


@dataclass
class Loss(ABC):
    """
    Class that holds the training and validation losses of its children calsses
    """

    _train: Metric
    _val: Metric
    accum_steps: Optional[int] = field(default=None)

    @abstractmethod
    def __call__(self, prediction: torch.Tensor, real: torch.Tensor) -> torch.Tensor: ...

    def train(self, prediction: torch.Tensor, real: torch.Tensor, with_grad: bool = False) -> bool:
        is_best: bool = self._train.set_tensor(self.__call__(prediction, real))

        if with_grad:
            loss = self._train.tensor / self.accum_steps if self.accum_steps is not None else self._train.tensor
            loss.backward()

        return is_best

    def val(self, prediction: torch.Tensor, real: torch.Tensor) -> bool:
        return self._val.set_tensor(self.__call__(prediction, real))

    def get(self, mode: str) -> float:
        if mode.lower().strip() in ["train", "training"]:
            return self._train.current
        elif mode.lower().strip() in ["val", "validation"]:
            return self._val.current
        else:
            raise AttributeError("Requested for invalid loss mode")


class L1Loss(Loss):
    def __init__(self, **kwargs) -> None:
        self.loss = torch.nn.L1Loss(**kwargs)

    def __call__(self, predicted: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.loss(predicted, real.squeeze(-1)).sum(-1).mean()


class IOU(Loss):
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def __call__(self, predicted: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        pred_arr: np.ndarray = (predicted <= 0.01).cpu().numpy()
        real_arr: np.ndarray = (real <= 0.01).cpu().numpy()
        if pred_arr.ndim >= 2:
            pred_arr = pred_arr.reshape(pred_arr.shape[0], -1)
        if real_arr.ndim >= 2:
            real_arr = real_arr.reshape(real_arr.shape[0], -1)

        pred_arr, real_arr = pred_arr >= 0.5, real_arr >= 0.5

        area_union = (pred_arr | real_arr).astype(np.float32).sum(axis=-1)
        area_intersect = (pred_arr & real_arr).astype(np.float32).sum(axis=-1)
        iou = np.divide(area_intersect, area_union, out=np.zeros_like(area_union), where=area_union != 0)
        return iou.mean()
