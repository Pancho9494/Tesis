import torch
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, Any, List
from torch.cuda.amp import GradScaler
from LIM.data.structures.cloud import Cloud
from LIM.data.structures.pair import Pair
from debug.decorators import identify_method
from debug.context import inspect_tensor
import concurrent.futures
from config import settings
from multimethod import multimethod


@dataclass
class Metric:
    """
    Class that tracks the best value for a loss
    """

    current: float = field(default=0.0)
    best: float = field(default=0.0)

    def set(self, new_value: float) -> bool:
        self.current = new_value
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
    scaler: GradScaler
    accum_steps: Optional[int] = field(default=None)
    device: torch.device = torch.device(settings.DEVICE)

    def __init__(self, scaler: GradScaler) -> None:
        self._train = Metric()
        self._val = Metric()
        self.scaler = scaler

    @abstractmethod
    def __call__(self, prediction: torch.Tensor, real: torch.Tensor) -> torch.Tensor: ...

    def train(self, sample: Any, with_grad: bool = False) -> bool:
        loss: torch.Tensor = self.__call__(sample)
        IS_BEST_STEP: bool = self._train.set(loss.item())

        if with_grad:
            loss /= self.accum_steps if self.accum_steps is not None else 1.0
            self.scaler.scale(loss).backward(retain_graph=True)

        return IS_BEST_STEP

    def val(self, sample: Any) -> bool:
        loss: torch.Tensor = self.__call__(sample)
        IS_BEST_STEP: bool = self._val.set(loss.item())
        return IS_BEST_STEP

    def get(self, mode: str) -> float:
        if mode.lower().strip() in ["train", "training"]:
            return self._train.current
        elif mode.lower().strip() in ["val", "validation"]:
            return self._val.current
        else:
            raise AttributeError("Requested for invalid loss mode")


class L1Loss(Loss):
    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["scaler"])
        self.loss = torch.nn.L1Loss(kwargs["accum_steps"])

    def __repr__(self) -> str:
        return "L1Loss"

    def __call__(self, predicted: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.loss(predicted, real.squeeze(-1)).sum(-1).mean()


class IOU(Loss):
    threshold: float

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["scaler"])
        self.threshold = kwargs["threshold"]

    def __repr__(self) -> str:
        return "IOU"

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


class MultiLoss(Loss):
    losses: List[Loss]

    def __init__(self, losses: List[Loss], **kwargs):
        super().__init__(kwargs["scaler"])
        self.losses = losses

    def __repr__(self) -> str:
        return f"MultiLoss({[loss for loss in self.losses]})"

    def __call__(self, pair: Pair) -> float:
        return sum([loss(pair) for loss in self.losses])

    def train(self, sample: Pair, with_grad: bool = False) -> bool:
        with torch.no_grad():
            sample.correspondences
            sample.source.pcd = sample.source.pcd.transform(sample.GT_tf_matrix)

        total: torch.Tensor = torch.tensor([0.0], device=self.device)
        for loss in self.losses:
            current_loss = loss(sample)
            loss._train.set(current_loss.item())
            total = total + current_loss

        IS_BEST_STEP: bool = self._train.set(total.item())

        if with_grad:
            total /= self.accum_steps if self.accum_steps is not None else 1.0
            self.scaler.scale(total).backward(retain_graph=True)
        return IS_BEST_STEP

    def val(self, sample: Pair) -> bool:
        with torch.no_grad():
            sample.correspondences
            sample.source.pcd = sample.source.pcd.transform(sample.GT_tf_matrix)
            total: torch.Tensor = torch.tensor([0.0], device=self.device)
            for loss in self.losses:
                current_loss = loss(sample)
                loss._val.set(current_loss.item())
                total = total + current_loss

        IS_BEST_STEP: bool = self._val.set(total.item())
        return IS_BEST_STEP


class CircleLoss(Loss):
    WEIGHT: float
    POS_RADIUS: float = 0.0375
    SAFE_RADIUS: float = 0.1
    POS_OPTIMAL: float = 0.1
    NEG_OPTIMAL: float = 1.4
    LOG_SCALE: int = 16
    POS_MARGIN: float = 0.1
    NEG_MARGIN: float = 1.4
    MAX_POINTS: int = 256

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["scaler"])
        self.WEIGHT = kwargs["weight"] if "weight" in kwargs else 1.0

    def __repr__(self):
        return f"CircleLoss(weight={self.WEIGHT})"

    def __call__(self, pair: Pair) -> torch.Tensor:
        if self.WEIGHT <= 0.0:
            return torch.tensor([0.0], device=self.device, requires_grad=True)

        sub_correspondence = pair.correspondences.matrix[
            torch.norm(
                input=(
                    pair.source.tensor[pair.correspondences.matrix[:, 0]]
                    - pair.target.tensor[pair.correspondences.matrix[:, 1]]
                ),
                dim=1,
            )
            < self.POS_RADIUS - 0.001
        ]
        if sub_correspondence.shape[0] > self.MAX_POINTS:
            sub_correspondence = sub_correspondence[torch.randperm(sub_correspondence.shape[0])[: self.MAX_POINTS]]

        src_idx, tgt_idx = sub_correspondence[:, 0], sub_correspondence[:, 1]
        src_pcd, tgt_pcd = pair.source.tensor[src_idx], pair.target.tensor[tgt_idx]
        # print(pair.source.features.grad_fn)
        src_feats, tgt_feats = pair.source.features[src_idx], pair.target.features[tgt_idx]
        coords_dist = torch.sqrt(self._square_distance(src_pcd[None, :, :], tgt_pcd[None, :, :]).squeeze(0))
        feats_dist = torch.sqrt(
            self._square_distance(src_feats[None, :, :], tgt_feats[None, :, :], normalised=True)
        ).squeeze(0)

        pos_mask = coords_dist < self.POS_RADIUS
        neg_mask = coords_dist > self.SAFE_RADIUS

        ## get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1) > 0) * (neg_mask.sum(-1) > 0)).detach()
        col_sel = ((pos_mask.sum(-2) > 0) * (neg_mask.sum(-2) > 0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float()  # mask the non-positive
        pos_weight = pos_weight - self.POS_OPTIMAL  # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach()

        neg_weight = feats_dist + 1e5 * (~neg_mask).float()  # mask the non-negative
        neg_weight = self.NEG_OPTIMAL - neg_weight  # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight).detach()

        lse_pos_row = torch.logsumexp(self.LOG_SCALE * (feats_dist - self.POS_MARGIN) * pos_weight, dim=-1)
        lse_pos_col = torch.logsumexp(self.LOG_SCALE * (feats_dist - self.POS_MARGIN) * pos_weight, dim=-2)

        lse_neg_row = torch.logsumexp(self.LOG_SCALE * (self.NEG_MARGIN - feats_dist) * neg_weight, dim=-1)
        lse_neg_col = torch.logsumexp(self.LOG_SCALE * (self.NEG_MARGIN - feats_dist) * neg_weight, dim=-2)

        loss_row = torch.nn.functional.softplus(lse_pos_row + lse_neg_row) / self.LOG_SCALE
        loss_col = torch.nn.functional.softplus(lse_pos_col + lse_neg_col) / self.LOG_SCALE

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2
        return self.WEIGHT * circle_loss

    def _square_distance(self, src, dst, normalised=False):
        """
        Calculate Euclid distance between each two points.
        Args:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Returns:
            dist: per-point square distance, [B, N, M]
        """
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += 2 if normalised else torch.sum(src**2, dim=-1)[:, :, None] + torch.sum(dst**2, dim=-1)[:, None, :]
        return torch.clamp(dist, min=1e-12, max=None)


class OverlapLoss(Loss):
    WEIGHT: float

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["scaler"])
        self.BCELoss = torch.nn.BCELoss(reduction="none")
        self.WEIGHT = kwargs["weight"] if "weight" in kwargs else 1.0

    def __repr__(self):
        return f"OverlapLoss(weight={self.WEIGHT})"

    def __call__(self, pair: Pair) -> torch.Tensor:
        if self.WEIGHT <= 0.0:
            return torch.tensor([0.0], device=self.device)

        ground_truth = torch.cat(
            tensors=(
                (
                    torch.zeros(pair.overlaps.src.shape[0])
                    .to(self.device)
                    .index_fill_(dim=0, index=pair.correspondences.source_indices, value=1)
                ),
                (
                    torch.zeros(pair.overlaps.target.shape[0])
                    .to(self.device)
                    .index_fill_(dim=0, index=pair.correspondences.target_indices, value=1)
                ),
            ),
            dim=0,
        )
        overlaps = torch.cat(tensors=(pair.overlaps.src, pair.overlaps.target), dim=0)
        return self.WEIGHT * self._weighted_BCELoss(overlaps, ground_truth)

    def _weighted_BCELoss(self, prediction: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        real: the indices that actually overlap
        prediction:
        """
        loss = self.BCELoss(prediction, real)
        N_NEGATIVE = real.sum() / real.shape[0]
        return torch.mean((torch.where(real >= 0.5, (1 - N_NEGATIVE) * loss, N_NEGATIVE * loss)))


class MatchabilityLoss(Loss):
    WEIGHT: float

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["scaler"])
        self.BCELoss = torch.nn.BCELoss(reduction="none")
        self.WEIGHT = kwargs["weight"] if "weight" in kwargs else 0.0

    def __repr__(self):
        return f"MatchabilityLoss(weight={self.WEIGHT})"

    # @identify_method(on=True)
    def __call__(self, pair: Pair) -> torch.Tensor:
        if self.WEIGHT <= 0.0:
            return torch.tensor([0.0], device=self.device, requires_grad=True)

        MATCHABILITY_RADIUS = 0.05
        saliencies = torch.cat(
            tensors=(
                pair.saliencies.src[src_idx := pair.correspondences.source_indices],
                pair.saliencies.target[tgt_idx := pair.correspondences.target_indices],
            )
        )
        scores = torch.matmul(pair.source.features[src_idx], pair.target.features[tgt_idx].transpose(0, 1))
        ground_truth = torch.cat(
            tensors=(
                torch.where(
                    condition=(
                        torch.norm(
                            pair.source.tensor[src_idx] - pair.target.tensor[tgt_idx][scores.argmax(1)], p=2, dim=1
                        )
                        < MATCHABILITY_RADIUS
                    ),
                    self=1.0,
                    other=0.0,
                ),
                torch.where(
                    condition=(
                        torch.norm(
                            pair.target.tensor[tgt_idx] - pair.source.tensor[src_idx][scores.argmax(0)], p=2, dim=1
                        )
                        < MATCHABILITY_RADIUS
                    ),
                    self=1.0,
                    other=0.0,
                ),
            ),
            dim=0,
        )

        return self.WEIGHT * self._weighted_BCELoss(saliencies, ground_truth)

    def _weighted_BCELoss(self, prediction: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        real: the indices that actually overlap
        prediction:
        """
        loss = self.BCELoss(prediction, real)
        N_NEGATIVE = real.sum() / real.shape[0]
        return torch.mean((torch.where(real >= 0.5, (1 - N_NEGATIVE) * loss, N_NEGATIVE * loss)))
