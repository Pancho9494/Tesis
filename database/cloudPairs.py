from dataclasses import dataclass, field
from pathlib import Path
import open3d as o3d
import numpy as np
from typing import Optional
import torch
import copy
from submodules.OverlapPredator.scripts.cal_overlap import get_overlap_ratio

@dataclass
class Cloud:
    path: Path
    _pcd: Optional[o3d.geometry.PointCloud] = field(default=None, repr=False)
    _arr: Optional[np.ndarray] = field(default=None, repr=False)
    _feat: Optional[torch.Tensor] = field(default=None, repr=False)
    
    def __len__(self) -> int:
        return self.arr.shape[0]
    
    @property
    def pcd(self) -> o3d.geometry.PointCloud:        
    # TODO: for some reason reading a point cloud prints "Extension = ply\nFormat = auto" which is pretty annoying
        if self._pcd is None:
            self._pcd = o3d.io.read_point_cloud(str(self.path))
        return self._pcd
    
    @pcd.setter
    def pcd(self, value: o3d.geometry.PointCloud) -> None:
        assert isinstance(value, o3d.geometry.PointCloud)
        self._pcd = value
    
    @property
    def arr(self) -> np.ndarray:
        if self._arr is None:
            self._arr = np.asfarray(self.pcd.points)
        return self._arr
    
    @arr.setter
    def arr(self, value: np.ndarray) -> None:
        assert isinstance(value, np.ndarray)
        self._arr = value
    
    @property
    def features(self) -> torch.Tensor:
        return self._feat
    
    @features.setter
    def features(self, value: torch.Tensor) -> None:
        assert torch.is_tensor(value)
        self._feat = value
        
    @property
    def index(self) -> int:
        return int(str(self.path.stem).replace("cloud_bin_", ""))
    
    def downsample(self, size: int) -> None:
        """
        Random downsample

        Args:
            size (int): The desired size of the cloud
        """
        if self._pcd is None or self._arr is None:
            raise RuntimeError("Object Cloud hasn't loaded its pcd yet")
        self._arr = self._arr[np.random.permutation(self._arr.shape[0])[:size]]
        self._pcd.points = o3d.utility.Vector3dVector(self._arr)
        
    def prob_subsample(self, size: int, overlap_score: torch.Tensor, saliency_score: torch.Tensor) -> 'Cloud':
        """
        Probabilistic downsample
        
        Selects the points of the cloud that are most important for registration, based on the given overlap and
        saliency scores

        Args:
            size (int): How many samples we eed
            overlap_score (torch.Tensor): The overlap scores for each point in the original cloud
            saliency_score (torch.Tensor): The saliency scores for each point in the original cloud

        Returns:
            Cloud: A subsampled copy of the original cloud
        """
        cloud_copy = copy.deepcopy(self)
        score = overlap_score * saliency_score
        temp_arr = torch.from_numpy(cloud_copy.arr)
        if (cloud_copy.arr.shape[0] <= size):
            return cloud_copy
        
        probabilities = (score / score.sum()).numpy().flatten()
        indices = np.random.choice(np.arange(temp_arr.shape[0]), size, replace=False, p=probabilities)
        
        cloud_copy.arr = cloud_copy._arr[indices]
        cloud_copy.features = cloud_copy._feat[indices]
        cloud_copy.pcd.points = o3d.utility.Vector3dVector(cloud_copy.arr)
        return cloud_copy

@dataclass
class FragmentPairs:
    """
    Utility class that holds fragment pairs, with the transform that aligns them and their overlap
    """
    src: Cloud
    target: Cloud
    transform: np.ndarray = field(compare=False, repr=False)
    overlap_ratio: float = field(default=0.0, compare=False, repr=False)
    
    def compute_overlap(self) -> None:
        temp = self.target.pcd
        temp.transform(self.transform)
        self.overlap_ratio = get_overlap_ratio(self.src.pcd, temp)
