# I don't like using sys.path.append, but given that OverlapPredator is not a package, its kinda hard not to
import sys

sys.path.append("./submodules/OverlapPredator/")
from pathlib import Path
from easydict import EasyDict as edict
import numpy as np
import torch
from dataclasses import dataclass, field

from submodules.OverlapPredator.models.architectures import KPFCNN
from submodules.OverlapPredator.lib.utils import load_config
from submodules.OverlapPredator.configs.models import architectures
from submodules.OverlapPredator.datasets.dataloader import collate_fn_descriptor
from submodules.OverlapPredator.lib.benchmark_utils import ransac_pose_estimation

from models.modelI import ModelI
from data.pairs import Pairs


@dataclass
class Config:
    """
    Utility class expected in PREDATOR's architecture, these values are originally read from a .yaml
    """

    deform_radius: float = field(default=5.0)
    first_subsampling_dl: float = field(default=0.025)
    conv_radius: float = field(default=2.5)
    architecture: list = field(default_factory=list)
    num_layers: int = field(default=4)
    n_points: int = field(default=1000)


class Predator(ModelI):
    """
    Wrapper for the PREDATOR model that follows the ModelI interface
    """

    model: torch.nn.Module
    config: Config
    _mode: str

    def __init__(self, mode: str) -> None:
        config = load_config(Path(f"./submodules/OverlapPredator/configs/test/{mode}.yaml"))
        config["architecture"] = architectures[mode]
        self.model = KPFCNN(edict(config))
        self.model.load_state_dict(
            torch.load(f"./customModels/weights/PREDATOR/{mode}.pth", weights_only=True)["state_dict"],
        )

        self.config = Config()
        self.config.architecture = architectures[mode]
        self._mode = mode

    def __repr__(self) -> str:
        # return self.model.__repr__()
        return f"Predator {self._mode}"

    def __call__(self, pair: Pairs) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            wrapper = (
                pair.src.arr,  # src_pcd
                pair.target.arr,  # tgt_pcd
                np.ones_like(pair.src.arr[:, :1]).astype(np.float32),  # src_feats
                np.ones_like(pair.target.arr[:, :1]).astype(np.float32),  # tgt_feats
                np.ones_like(pair.truth[:3, :3]).astype(np.float32),  # rot
                np.ones_like(pair.truth[:, 3]).astype(np.float32),  # trans
                torch.ones(1, 2).long(),  # correspondences
                pair.src.arr,  # src_raw
                pair.target.arr,  # tgt_raw
                torch.ones(1),  # samplee
            )
            batched_input = collate_fn_descriptor(
                [wrapper],
                self.config,
                neighborhood_limits=[int(np.ceil(4 / 3 * np.pi * (self.config.deform_radius + 1) ** 3))] * 5,
            )
            feats, overlaps, saliencies = self.model(batched_input)
            len_src = batched_input["stack_lengths"][0][0]
            pair.src.features, pair.target.features = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
            src_overlap, src_saliency = overlaps[:len_src].detach().cpu(), saliencies[:len_src].detach().cpu()
            tgt_overlap, tgt_saliency = overlaps[len_src:].detach().cpu(), saliencies[len_src:].detach().cpu()

            src_overlap_cloud = pair.src.prob_subsample(self.config.n_points, src_overlap, src_saliency)
            target_overlap_cloud = pair.target.prob_subsample(self.config.n_points, tgt_overlap, tgt_saliency)

            # THere's something weird here, when the datased is loaded inside the threeDLoMatch class the ground truth
            # transformation is meant to transform the target to th be position of the source, but the PREDATOR
            # prediction is trasforming the source to the target. So, in the predator demo.py file the
            # ransac_pose_estimation function receives(src_pcd, tgt_pcd, src_feats, tgt_feats), but here the order
            # between src and target is flipped, so the transformation is consistent
            transform = ransac_pose_estimation(
                target_overlap_cloud.arr,
                src_overlap_cloud.arr,
                target_overlap_cloud.features,
                src_overlap_cloud.features,
                mutual=False,
            )
            return transform
