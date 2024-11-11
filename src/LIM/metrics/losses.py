import torch
import numpy as np

torch.set_printoptions(precision=7)


class L1Loss:
    def __init__(self, **kwargs) -> None:
        self.loss = torch.nn.L1Loss(**kwargs)

    def __call__(self, predicted: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.loss(predicted, real).sum(-1).mean()


class IOU:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def __call__(self, predicted: torch.Tensor, real: torch.Tensor) -> float:
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
