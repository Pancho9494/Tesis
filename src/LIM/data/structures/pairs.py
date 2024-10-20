from dataclasses import dataclass, field
import open3d as o3d
import numpy as np
from LIM.data.cloud import Cloud
from typing import Union, Optional
import copy


@dataclass
class Pairs:
    """
    Utility class that holds cloud pairs, with the transform that aligns them and their overlap
    """

    src: Cloud
    target: Cloud
    # TODO: we should hold both the ground truth transformation and the predicted transformation, or maybe just the GT
    # and we work with predictions just as an argument
    truth: np.ndarray = field(compare=False, repr=False)  # somehow this is completely wrong???
    prediction: Union[np.ndarray, None] = field(default=None, compare=False, repr=False)
    _overlap: Optional[float] = field(default=None)

    def __repr__(self) -> str:
        return f"Pair({self.src.path.stem}, {self.target.path.stem})"

    def _compute_overlap(self, transform: np.ndarray) -> float:
        temp = self.target.pcd
        temp.transform(transform)
        return self._get_overlap_ratio(self.src.pcd, temp)

    @property
    def overlap(self) -> float:
        if self._overlap is None:
            self._overlap = self.GT_overlap()
        return self._overlap

    def GT_overlap(self) -> float:
        return self._compute_overlap(self.truth)

    def pred_overlap(self) -> float:
        if self.prediction is None:
            return 0.0
        return self._compute_overlap(self.prediction)

    def show(self, num_steps: int = 1) -> None:
        def _animate_transform(vis) -> None:
            nonlocal prediction
            for i in range(num_steps + 1):
                alpha = i / num_steps
                intermediate_matrix = (1 - alpha) * np.eye(4) + alpha * self.prediction
                prediction.pcd.transform(intermediate_matrix)
                vis.update_geometry(prediction.pcd)
                vis.poll_events()
                vis.update_renderer()
                prediction.pcd.transform(np.linalg.inv(intermediate_matrix))

        self.src.paint([1, 0.706, 0])

        ground_truth = copy.deepcopy(self.target)
        ground_truth.paint([1, 0, 0])
        ground_truth.pcd.transform(self.truth)

        prediction = copy.deepcopy(self.target)
        prediction.paint([0, 0.651, 0.929])

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.add_geometry(self.src.pcd)
        vis.add_geometry(prediction.pcd)
        vis.add_geometry(ground_truth.pcd)
        vis.register_key_callback(ord("A"), _animate_transform)
        vis.run()

    def _get_overlap_ratio(self, source, target, threshold=0.03):
        """
        We compute overlap ratio from source point cloud to target point cloud
        """
        pcd_tree = o3d.geometry.KDTreeFlann(target)

        match_count = 0
        for i, point in enumerate(source.points):
            [count, _, _] = pcd_tree.search_radius_vector_3d(point, threshold)
            if count != 0:
                match_count += 1

        overlap_ratio = match_count / len(source.points)
        return overlap_ratio
