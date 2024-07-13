from dataclasses import dataclass, field
import open3d as o3d
import numpy as np
from submodules.OverlapPredator.scripts.cal_overlap import get_overlap_ratio
from database.cloud import Cloud

@dataclass
class Pairs:
    """
    Utility class that holds cloud pairs, with the transform that aligns them and their overlap
    """
    src: Cloud
    target: Cloud
    transform: np.ndarray = field(compare=False, repr=False)
    overlap_ratio: float = field(default=0.0, compare=False, repr=False)
    
    def compute_overlap(self) -> None:
        temp = self.target.pcd
        temp.transform(self.transform)
        self.overlap_ratio = get_overlap_ratio(self.src.pcd, temp)
        
    def show(self, apply_transform: bool, rotation_speed: float = 2.0) -> None:
        def _rotate_view(vis):
            ctr = vis.get_view_control()
            ctr.rotate(rotation_speed, 0.0)
            return False
        
        self.src.paint(np.array([1, 0.706, 0]))
        self.target.paint(np.array([0, 0.651, 0.929]))
        
        if apply_transform:
            self.src.pcd.transform(self.transform)
        
        o3d.visualization.draw_geometries_with_animation_callback([self.src.pcd, self.target.pcd], _rotate_view)
