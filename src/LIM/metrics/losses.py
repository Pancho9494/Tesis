import aim
import torch
import numpy as np
from typing import Protocol, Tuple, Optional
from LIM.data.structures.pair import Pair
from LIM.metrics import Loss
from debug.decorators import identify_method


class TrainerStateProtocol(Protocol):
    class CurrentProtocol(Protocol):
        iteration: int
        epoch: int
        step: int

    tracker: aim.Run
    current: CurrentProtocol


class L1Loss(Loss):
    def __init__(self, trainer_state: TrainerStateProtocol, reduction: str) -> None:
        super().__init__(trainer_state, also_track=["average"])
        self.reduction = reduction
        self.loss = torch.nn.L1Loss(reduction=self.reduction)

    def __repr__(self) -> str:
        return f"L1Loss(reduction={self.reduction})"

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        predicted, real = sample
        real = real.T
        return self.loss(predicted, real).sum(-1).mean()


class IOU(Loss):
    THRESHOLD: float

    def __init__(self, trainer_state: TrainerStateProtocol, threshold: float) -> None:
        super().__init__(trainer_state, y0to1=True, also_track=["average"])
        self.THRESHOLD = threshold

    def __repr__(self) -> str:
        return f"IOU(threshold={self.THRESHOLD})"

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        predicted, real = sample
        occ_pred: np.ndarray = (predicted <= 0.01).cpu().numpy()
        occ_true: np.ndarray = (real <= 0.01).cpu().numpy().T

        if occ_true.ndim >= 2:
            occ_true = occ_true.reshape(occ_true.shape[0], -1)
        if occ_pred.ndim >= 2:
            occ_pred = occ_pred.reshape(occ_pred.shape[0], -1)

        occ_pred = occ_pred >= self.THRESHOLD
        occ_true = occ_true >= self.THRESHOLD

        union = np.logical_or(occ_true, occ_pred).sum(axis=1)
        intersection = np.logical_and(occ_true, occ_pred).sum(axis=1)
        iou_per_sample = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=union != 0,
        )
        return iou_per_sample.mean()


class OverlapLoss(Loss):
    weight: float
    current_overlap_score: Optional[torch.Tensor] = None

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

        return self.weight * self._weighted_BCELoss(self.current_overlap_score, ground_truth)

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
    current_saliency_score: Optional[torch.Tensor] = None

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
        cat_features = torch.cat(
            (pair.source.features, pair.target.features),
            dim=torch.argmax(torch.tensor(pair.source.points.shape)).item(),
        )
        scores = torch.matmul(cat_features[:len_src][src_idx], cat_features[len_src:][tgt_idx].transpose(0, 1))
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

        src_saliency_scores = self.current_saliency_score[:len_src]
        src_saliency_scores = src_saliency_scores[src_idx]
        tgt_saliency_scores = self.current_saliency_score[len_src:]
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

        cat_features = torch.cat(
            (pair.source.features, pair.target.features),
            dim=torch.argmax(torch.tensor(pair.source.points.shape)).item(),
        )
        len_src = len(pair.source.first.points)
        src_idx, tgt_idx = sub_correspondence[:, 0], sub_correspondence[:, 1]
        src_pcd, tgt_pcd = pair.source.first.points[src_idx], pair.target.first.points[tgt_idx]
        src_feats, tgt_feats = cat_features[:len_src][src_idx], cat_features[len_src:][tgt_idx]

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
