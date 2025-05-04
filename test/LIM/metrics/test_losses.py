from LIM.metrics.losses import OverlapLoss
from LIM.data.structures.pair import Pair, PCloud
from LIM.models.trainer import RunState
import numpy as np


def test_overlap_loss() -> None:
    loss = OverlapLoss(trainer_state=RunState(), weight=1.0)
    coordinates = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    identity = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    pair = Pair(
        source=PCloud.from_arr(coordinates),
        target=PCloud.from_arr(coordinates),
        GT_tf_matrix=identity,
    )
    loss(pair)
    return
