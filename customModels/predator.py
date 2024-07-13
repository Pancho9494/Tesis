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

from customModels.modelI import ModelI
from database.cloudPairs import FragmentPairs

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
    
    def __init__(self, mode: str) -> None:
        path_to_test_config = Path(f"./submodules/OverlapPredator/configs/test/{mode}.yaml") 
        config = load_config(path_to_test_config)
        config['architecture'] = architectures[mode]
        self.model = KPFCNN(edict(config))
        self.model.load_state_dict(torch.load(f"./customModels/weights/PREDATOR/{mode}.pth")['state_dict'])
            
        self.config = Config()
        self.config.architecture = architectures[mode]
    
    def __repr__(self) -> str:
        return self.model.__repr__()
 
    def __call__(self, pair: FragmentPairs) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            wrapper = (
                pair.src.arr,                                               # src_pcd
                pair.target.arr,                                            # tgt_pcd
                np.ones_like(pair.src.arr[:, :1]).astype(np.float32),       # src_feats
                np.ones_like(pair.target.arr[:, :1]).astype(np.float32),    # tgt_feats
                pair.transform[:3, :3],                                     # rot
                pair.transform[:, 3],                                       # trans
                torch.ones(1,2).long(),                                     # correspondences
                pair.src.arr,                                               # src_raw
                pair.target.arr,                                            # tgt_raw
                torch.ones(1)                                               # samplee
            )
            batched_input = collate_fn_descriptor(
                [wrapper],
                self.config, 
                neighborhood_limits=[int(np.ceil(4 / 3 * np.pi * (self.config.deform_radius + 1) ** 3))] * 5
            )
            feats, overlaps, saliencies = self.model(batched_input)
            len_src = batched_input['stack_lengths'][0][0]
            pair.src.features, pair.target.features = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
            src_overlap, src_saliency = overlaps[:len_src].detach().cpu(), saliencies[:len_src].detach().cpu()
            tgt_overlap, tgt_saliency = overlaps[len_src:].detach().cpu(), saliencies[len_src:].detach().cpu()
            
            src_overlap_cloud = pair.src.prob_subsample(self.config.n_points, src_overlap, src_saliency)
            target_overlap_cloud = pair.target.prob_subsample(self.config.n_points, tgt_overlap, tgt_saliency)
            
            transform = ransac_pose_estimation(
                src_overlap_cloud.arr, target_overlap_cloud.arr, 
                src_overlap_cloud.features, target_overlap_cloud.features, 
                mutual=False
            )
            
            return transform
            
        
 