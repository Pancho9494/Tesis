import functools
import torch
from tqdm import tqdm
from config.config import settings
from LIM.models.modelI import Model
from LIM.data.sets.datasetI import CloudDatasetsI
from LIM.data.structures.pcloud import Downsampler, PCloud
import open3d as o3d
import open3d.pipelines.registration as o3d_reg
import numpy as np
import copy


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


class Evaluator:
    device: torch.device = torch.device(settings.DEVICE)
    model: Model
    dataset: CloudDatasetsI

    def __init__(self, model: Model, dataset: CloudDatasetsI) -> None:
        self.model = model.to(self.device)
        self.model.load("latest")
        self.dataset = dataset
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=settings.TESTER.BATCH_SIZE,
            shuffle=False,
            collate_fn=functools.partial(
                self.dataset.collate_fn,
                tf_pipeline=[],
            ),
        )

    def __call__(self) -> None:
        self.model.eval()
        downsampler = Downsampler(size=settings.TESTER.DOWNSAMPLE_SIZE, mode=Downsampler.Mode.PROBABILISTIC)
        ransac = RANSAC(
            distance_threshold=settings.TESTER.RANSAC.DISTANCE_THRESHOLD,
            similarity_threshold=settings.TESTER.RANSAC.SIMILARITY_THRESHOLD,
            max_iterations=settings.TESTER.RANSAC.MAX_ITERATIONS,
            n_correspondences=3,
        )
        inlier_ratio = InlierRatio(distance_threshold=0.1)
        with torch.no_grad():
            for sample in tqdm(self.test_loader):
                raw = copy.deepcopy(sample)
                sample.correspondences
                sample = self.model(sample)
                tf_matrix = ransac(
                    downsampler(sample.source, scores=sample.overlaps.src * sample.saliencies.src),
                    downsampler(sample.target, scores=sample.overlaps.target * sample.saliencies.target),
                )
                raw.overlaps = sample.overlaps
                print(inlier_ratio(sample.source, sample.target))
                raw.show(tf_matrix)
