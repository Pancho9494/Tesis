import copy
import functools

import numpy as np
import open3d.pipelines.registration as o3d_reg
import polars as pl
import torch

import LIM.log as log
from config.config import settings
from LIM.data.sets.datasetI import CloudDatasetsI
from LIM.data.structures.pair import Correspondences, Pair
from LIM.data.structures.pcloud import Downsampler, PCloud
from LIM.models.modelI import Model


class RANSAC:
    DISTANCE_THRESHOLD: float
    SIMILARITY_THRESHOLD: float
    MAX_ITERATIONS: int
    N_CORRESPONDENCES: int

    def __init__(
        self,
        distance_threshold: float,
        similarity_threshold: float,
        max_iterations: int,
        n_correspondences: int = 3,
    ) -> None:
        self.DISTANCE_THRESHOLD = distance_threshold
        self.SIMILARITY_THRESHOLD = similarity_threshold
        self.MAX_ITERATIONS = max_iterations
        self.N_CORRESPONDENCES = n_correspondences

    def __call__(self, source: PCloud, target: PCloud) -> np.ndarray:
        result = o3d_reg.registration_ransac_based_on_feature_matching(
            source=source.pcd.to_legacy(),
            target=target.pcd.to_legacy(),
            source_feature=source.o3d_features,
            target_feature=target.o3d_features,
            mutual_filter=False,
            max_correspondence_distance=self.DISTANCE_THRESHOLD,
            estimation_method=o3d_reg.TransformationEstimationPointToPoint(False),
            ransac_n=self.N_CORRESPONDENCES,
            checkers=[
                o3d_reg.CorrespondenceCheckerBasedOnEdgeLength(similarity_threshold=self.SIMILARITY_THRESHOLD),
                o3d_reg.CorrespondenceCheckerBasedOnDistance(distance_threshold=self.DISTANCE_THRESHOLD),
            ],
            criteria=o3d_reg.RANSACConvergenceCriteria(max_iteration=self.MAX_ITERATIONS),
        )
        return result.transformation


class InlierRatio:
    DISTANCE_THRESHOLD: float
    scores: torch.Tensor | None = None

    def __init__(self, distance_threshold: float) -> None:
        self.DISTANCE_THRESHOLD = distance_threshold

    def __call__(self, source: PCloud, target: PCloud) -> tuple[float, float]:
        self.scores = torch.matmul(source.features, target.features.T)
        return (
            self._without_mutual_check(source, target),
            self._with_mutual_check(source, target),
        )

    def _without_mutual_check(self, source: PCloud, target: PCloud) -> float:
        scores = torch.matmul(source.features, target.features.T) if self.scores is None else self.scores
        dist = torch.norm(source.points - target.points[scores.argmax(dim=-1)], dim=1)
        return (dist < self.DISTANCE_THRESHOLD).float().mean().item()

    def _with_mutual_check(self, source: PCloud, target: PCloud) -> float:
        scores = torch.matmul(source.features, target.features.T) if self.scores is None else self.scores
        rows, cols = np.where(self._mutual_selection(scores))
        dist = torch.norm(source.points[rows] - target.points[cols], dim=1)
        return (dist < self.DISTANCE_THRESHOLD).float().mean().item()

    def _mutual_selection(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column

        Args: np.array()
            score_mat:  [B,N,N]
        Return:
            mutuals:    [B,N,N]
        """
        scores = scores[None, :, :].cpu().numpy()
        mutuals = np.zeros_like(scores)
        for idx in range(scores.shape[0]):
            c_mat = scores[idx]
            flag_row = np.zeros_like(c_mat)
            flag_column = np.zeros_like(c_mat)

            max_along_row = np.argmax(c_mat, 1)[:, None]
            max_along_col = np.argmax(c_mat, 0)[None, :]

            np.put_along_axis(flag_row, max_along_row, values=1, axis=1)
            np.put_along_axis(flag_column, max_along_col, values=1, axis=0)

            mutuals[idx] = (flag_row.astype(np.bool)) & (flag_column.astype(np.bool))
        return mutuals.astype(np.bool)[0]


class RelativeRotationError:
    def __call__(self, real_T: np.ndarray, pred_T: np.ndarray) -> float:
        real_R, pred_R = real_T[:3, :3], pred_T[:3, :3]
        trace = np.clip(np.trace(pred_R.T @ real_R), -1.0, 1.0)
        return np.arccos((trace - 1) / 2)


class RelativeTranslationError:
    def __call__(self, real_T: np.ndarray, pred_T: np.ndarray) -> float:
        real_t, pred_t = real_T[:3, 3], pred_T[:3, 3]
        return np.linalg.norm(pred_t - real_t)


class TransformationError:
    def __call__(self) -> float: ...


class RootMeanSquaredError:
    def __call__(
        self, source: PCloud, target: PCloud, tf_matrix: np.ndarray, correspondences: Correspondences
    ) -> float:
        if len(correspondences.matrix) == 0:
            return float("inf")  # RMSE < tau will always be False
        TX = source.transform(tf_matrix).arr[correspondences.source_indices.cpu().numpy()]
        Y = target.arr[correspondences.target_indices.cpu().numpy()]
        return float(np.sqrt(np.mean(np.sum((TX - Y) ** 2, axis=1))))


class RegistrationRecall:
    tau: float

    def __init__(self, tau: float) -> None:
        self.tau = tau

    def __call__(self, rmse_list: list[float]) -> float:
        RR = 0
        for rmse in rmse_list:
            RR += 1 if rmse < self.tau else 0
        return RR / len(rmse_list)


class Evaluator:
    device: torch.device = torch.device(settings.DEVICE)
    model: Model
    dataset: CloudDatasetsI

    def __init__(self, model: Model, dataset: CloudDatasetsI) -> None:
        self.model = model.to(self.device)
        self.model.load(
            run=f"{settings.TRAINER.BACKUP_DIR}/{self.model.__class__.__name__}/{settings.TRAINER.DATED}",
            suffix="latest",
        )
        self.dataset = dataset
        log.info(f"Loaded {self.dataset}")
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=settings.TESTER.BATCH_SIZE,
            shuffle=False,
            collate_fn=functools.partial(self.dataset.collate_fn),
        )

    def __call__(self) -> None:
        self.model.eval()
        ransac = RANSAC(
            distance_threshold=settings.TESTER.RANSAC.DISTANCE_THRESHOLD,
            similarity_threshold=settings.TESTER.RANSAC.SIMILARITY_THRESHOLD,
            max_iterations=settings.TESTER.RANSAC.MAX_ITERATIONS,
            n_correspondences=3,
        )
        RRE = RelativeRotationError()
        RTE = RelativeTranslationError()
        RMSE = RootMeanSquaredError()
        TAU = 0.2
        RR = RegistrationRecall(tau=TAU)

        for N_RANSAC in [250, 500, 1000, 2500, 5000]:
            overlap_list = []
            rmses = []
            rres = []
            rtes = []
            downsampler = Downsampler(size=N_RANSAC, mode=Downsampler.Mode.PROBABILISTIC)
            with torch.no_grad():
                sample: Pair
                for sample in self.test_loader:
                    try:
                        raw = copy.deepcopy(sample)
                        sample.correspondences
                        sample, overlaps, saliencies = self.model(sample)
                        source, target = sample

                        N = source.shape[0]
                        src_overlaps, src_saliencies = overlaps[:N], saliencies[:N]
                        tgt_overlaps, tgt_saliencies = overlaps[N:], saliencies[N:]

                        src_down, src_indices = downsampler(source, scores=src_overlaps * src_saliencies)
                        tgt_down, tgt_indices = downsampler(target, scores=tgt_overlaps * tgt_saliencies)

                        tf_matrix = ransac(src_down, tgt_down)
                        rre = max(0, RRE(sample.GT_tf_matrix, tf_matrix))
                        rte = max(0, RTE(sample.GT_tf_matrix, tf_matrix))

                        src_reverse_mapping = {orig.item(): new for new, orig in enumerate(src_indices)}
                        tgt_reverse_mapping = {orig.item(): new for new, orig in enumerate(tgt_indices)}

                        corr_down = []
                        for s_idx, t_idx in zip(
                            sample._correspondences.source_indices, sample._correspondences.target_indices
                        ):
                            s_idx, t_idx = s_idx.item(), t_idx.item()
                            if (s_idx in src_reverse_mapping) and (t_idx in tgt_reverse_mapping):
                                corr_down.append(
                                    [
                                        src_reverse_mapping[s_idx],
                                        tgt_reverse_mapping[t_idx],
                                    ]
                                )

                        rmse = max(
                            RMSE(
                                src_down,
                                tgt_down,
                                # sample.GT_tf_matrix,
                                tf_matrix,
                                Correspondences(
                                    torch.tensor(corr_down, device=self.device),
                                ),
                            ),
                            0,
                        )
                        rmses.append(rmse)
                        overlap_list.append(sample._overlap)

                        log.info(f"Overlap={sample._overlap:2.2f}, RRE={rre:2.2f}, RTE={rte:2.2f}, RMSE={rmse:2.2f}")
                        rres.append(rre)
                        rtes.append(rte)
                        # if rre < 1 or rte < 1:
                        raw.show(tf_matrix)
                    except RuntimeError:
                        log.warn("CUDA out of memory error!")
                        continue
            rre = f"{str(round(np.mean(rres), 2)).replace('.', '_')}"
            rte = f"{str(round(np.mean(rtes), 2)).replace('.', '_')}"
            rr = f"{str(round(RR([value for value in rmses if value != float('inf')]), 2) * 100).replace('.', '_')}"
            log.info(f"N_RANSAC={N_RANSAC}, TAU={TAU}, Average RRE={rre}, Average RTE={rte}, RR={rr}%")
            df = (
                pl.DataFrame(
                    {
                        "Overlap": overlap_list,
                        "RRE": rres,
                        "RTE": rtes,
                        "RMSE": rmses,
                    }
                )
                .sort(pl.col("Overlap"))
                .with_columns(pl.col("RMSE").replace({np.inf: None}))
            )
            log.info(df)
            # df.write_csv(f"./NRANSAC__{N_RANSAC}__TAU__{TAU}__RRE__{rre}__RTE__{rte}__RR__{rr}.csv")
