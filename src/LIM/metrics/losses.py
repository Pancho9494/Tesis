import aim
import torch
import numpy as np
from typing import Protocol, Tuple
from LIM.data.structures.pair import Pair
from LIM.metrics import Loss
from debug.decorators import identify_method
from sklearn.metrics import precision_recall_fscore_support


class TrainerStateProtocol(Protocol):
    class CurrentProtocol(Protocol):
        epoch: int
        step: int

    tracker: aim.Run
    current: CurrentProtocol


class L1Loss(Loss):
    def __init__(self, trainer_state: TrainerStateProtocol, reduction: str) -> None:
        super().__init__(trainer_state)
        self.reduction = reduction
        self.loss = torch.nn.L1Loss(self.reduction)

    def __repr__(self) -> str:
        return f"L1Loss(reduction={self.reduction})"

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        predicted, real = sample
        return self.loss(predicted, real.squeeze(-1)).sum(-1).mean()


class IOU(Loss):
    THRESHOLD: float

    def __init__(self, trainer_state: TrainerStateProtocol, threshold: float) -> None:
        super().__init__(trainer_state)
        self.THRESHOLD = threshold

    def __repr__(self) -> str:
        return f"IOU(threshold={self.THRESHOLD})"

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        predicted, real = sample
        pred_arr: np.ndarray = (predicted <= 0.01).cpu().numpy()
        real_arr: np.ndarray = (real <= 0.01).cpu().numpy()
        if pred_arr.ndim >= 2:
            pred_arr = pred_arr.reshape(pred_arr.shape[0], -1)
        if real_arr.ndim >= 2:
            real_arr = real_arr.reshape(real_arr.shape[0], -1)

        pred_arr, real_arr = pred_arr >= self.THRESHOLD, real_arr >= self.THRESHOLD

        area_union = (pred_arr | real_arr).astype(np.float32).sum(axis=-1)
        area_intersect = (pred_arr & real_arr).astype(np.float32).sum(axis=-1)
        iou = np.divide(area_intersect, area_union, out=np.zeros_like(area_union), where=area_union != 0)
        return iou.mean()


class OverlapLoss(Loss):
    weight: float

    def __init__(self, trainer_state: TrainerStateProtocol, weight: float) -> None:
        super().__init__(trainer_state, also_track=["average"])
        self.BCELoss = torch.nn.BCELoss(reduction="none")
        self.weight = weight

    def __repr__(self):
        return f"OverlapLoss(weight={self.weight})"

    @identify_method
    def __call__(self, pair: Pair) -> torch.Tensor:
        if self.weight <= 0.0:
            return torch.tensor([0.0], device=self.device)

        ground_truth = torch.cat(
            tensors=(
                (
                    torch.zeros(pair.source.first.points.shape[0])
                    .to(self.device)
                    .index_fill_(dim=0, index=pair.correspondences.source_indices, value=1)
                ),
                (
                    torch.zeros(pair.target.first.points.shape[0])
                    .to(self.device)
                    .index_fill_(dim=0, index=pair.correspondences.target_indices, value=1)
                ),
            ),
            dim=0,
        )

        return self.weight * self._weighted_BCELoss(pair.overlaps.mix, ground_truth)

    def _weighted_BCELoss(self, prediction: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        real: the indices that actually overlap
        prediction:
        """
        loss = self.BCELoss(prediction, real)
        N_NEGATIVE = real.sum() / real.shape[0]
        return torch.mean((torch.where(real >= 0.5, (1 - N_NEGATIVE) * loss, N_NEGATIVE * loss)))


class MatchabilityLoss(Loss):
    weight: float = 0.0

    def __init__(self, trainer_state: TrainerStateProtocol, weight: float) -> None:
        super().__init__(trainer_state, also_track=["average"])
        self.BCELoss = torch.nn.BCELoss(reduction="none")
        self.weight = weight

    def __repr__(self):
        return f"MatchabilityLoss(weight={self.weight})"

    @identify_method
    def __call__(self, pair: Pair) -> torch.Tensor:
        if self.weight <= 0.0:
            return torch.tensor([0.0], device=self.device, requires_grad=True)

        MATCHABILITY_RADIUS = 0.05
        src_idx = pair.correspondences.source_indices
        tgt_idx = pair.correspondences.target_indices

        len_src = len(pair.source.first.points)
        scores = torch.matmul(
            pair.mix.features[:len_src][src_idx], pair.mix.features[len_src:][tgt_idx].transpose(0, 1)
        )
        ground_truth = torch.cat(
            tensors=(
                torch.where(
                    condition=(
                        torch.norm(
                            pair.source.first.points[src_idx] - pair.target.first.points[tgt_idx][scores.argmax(1)],
                            p=2,
                            dim=1,
                        )
                        < MATCHABILITY_RADIUS
                    ),
                    self=1.0,
                    other=0.0,
                ),
                torch.where(
                    condition=(
                        torch.norm(
                            pair.target.first.points[tgt_idx] - pair.source.first.points[src_idx][scores.argmax(0)],
                            p=2,
                            dim=1,
                        )
                        < MATCHABILITY_RADIUS
                    ),
                    self=1.0,
                    other=0.0,
                ),
            ),
            dim=0,
        )

        src_saliency_scores = pair.saliencies.mix[:len_src]
        src_saliency_scores = src_saliency_scores[src_idx]
        tgt_saliency_scores = pair.saliencies.mix[len_src:]
        tgt_saliency_scores = tgt_saliency_scores[tgt_idx]
        scores_saliency = torch.cat((src_saliency_scores, tgt_saliency_scores))

        return self.weight * self._weighted_BCELoss(scores_saliency, ground_truth)

    def _weighted_BCELoss(self, prediction: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        real: the indices that actually overlap
        prediction:
        """
        loss = self.BCELoss(prediction, real)
        N_NEGATIVE = real.sum() / real.shape[0]
        return torch.mean((torch.where(real >= 0.5, (1 - N_NEGATIVE) * loss, N_NEGATIVE * loss)))


global coords_dist, feats_dist


class CircleLoss(Loss):
    weight: float
    POS_RADIUS: float = 0.0375
    SAFE_RADIUS: float = 0.1
    POS_OPTIMAL: float = 0.1
    NEG_OPTIMAL: float = 1.4
    LOG_SCALE: int = 24
    POS_MARGIN: float = 0.1
    NEG_MARGIN: float = 1.4
    MAX_POINTS: int = 256

    def __init__(self, trainer_state: TrainerStateProtocol, weight: float) -> None:
        super().__init__(trainer_state, also_track=["average"])
        self.weight = weight

    def __repr__(self):
        return f"CircleLoss(weight={self.weight})"

    @identify_method
    def __call__(self, pair: Pair) -> torch.Tensor:
        if self.weight <= 0.0:
            return torch.tensor([0.0], device=self.device, requires_grad=True)

        sub_correspondence = pair.correspondences.matrix[
            torch.norm(
                input=(
                    pair.source.first.points[pair.correspondences.matrix[:, 0]]
                    - pair.target.first.points[pair.correspondences.matrix[:, 1]]
                ),
                dim=1,
            )
            < self.POS_RADIUS - 0.001
        ]
        if sub_correspondence.shape[0] > self.MAX_POINTS:
            sub_correspondence = sub_correspondence[torch.randperm(sub_correspondence.shape[0])[: self.MAX_POINTS]]

        len_src = len(pair.source.first.points)
        src_idx, tgt_idx = sub_correspondence[:, 0], sub_correspondence[:, 1]
        src_pcd, tgt_pcd = pair.source.first.points[src_idx], pair.target.first.points[tgt_idx]
        src_feats, tgt_feats = pair.mix.features[:len_src][src_idx], pair.mix.features[len_src:][tgt_idx]

        global coords_dist, feats_dist
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
        loss = self.weight * circle_loss
        return loss

    def _square_distance(self, src, dst, normalised=False):
        """
        Calculate Euclid distance between each two points.
        Args:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Returns:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        if normalised:
            dist += 2
        else:
            dist += torch.sum(src**2, dim=-1)[:, :, None]
            dist += torch.sum(dst**2, dim=-1)[:, None, :]

        dist = torch.clamp(dist, min=1e-12, max=None)
        return dist


class FeatureMatchRecall(Loss):
    MAX_POINTS: int = 256
    POS_RADIUS: float = 0.0375

    def __init__(self, trainer_state: TrainerStateProtocol) -> None:
        super().__init__(trainer_state, y0to1=True, also_track=["average"])

    def __repr__(self) -> str:
        return "FeatureMatchRecal()"

    @identify_method
    def __call__(self, sample: Pair) -> torch.Tensor:
        global coords_dist, feats_dist
        pos_mask = coords_dist < self.POS_RADIUS
        n_gt_pos = (pos_mask.sum(-1) > 0).float().sum() + 1e-12
        _, sel_idx = torch.min(feats_dist, -1)
        sel_dist = torch.gather(coords_dist, dim=-1, index=sel_idx[:, None])[pos_mask.sum(-1) > 0]
        n_pred_pos = (sel_dist < self.POS_RADIUS).float().sum()
        recall = n_pred_pos / n_gt_pos
        if torch.isnan(recall).any():
            print("?")
        return recall

    def _square_distance(self, src, dst, normalised=False):
        """
        Calculate Euclid distance between each two points.
        Args:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Returns:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        if normalised:
            dist += 2
        else:
            dist += torch.sum(src**2, dim=-1)[:, :, None]
            dist += torch.sum(dst**2, dim=-1)[:, None, :]

        dist = torch.clamp(dist, min=1e-12, max=None)
        return dist
