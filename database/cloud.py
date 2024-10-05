from dataclasses import dataclass, field
import numpy as np
import open3d as o3d
from pathlib  import Path
from typing import Optional, Union, List
import torch
import copy
import matplotlib as mpl


@dataclass
class Cloud:
    path: Path
    _pcd: Optional[o3d.geometry.PointCloud] = field(default=None, repr=False)
    _arr: Optional[np.ndarray] = field(default=None, repr=False)
    _feat: Optional[torch.Tensor] = field(default=None, repr=False)
    
    def __len__(self) -> int:
        if self.arr is not None:
            return self.arr.shape[0]
        return 0
    
    @property
    def pcd(self) -> o3d.geometry.PointCloud:
        """
        We read the point clouds only when the user asks for them
        
        TODO: for some reason reading a point cloud prints "Extension = ply\nFormat = auto" which is pretty annoying
        """
        if self._pcd is None:
            if self.path.suffix.lower() in [".npz", ".npy"]:
                self._pcd = o3d.geometry.PointCloud()
                data = np.load(self.path)
                self._pcd.points = o3d.utility.Vector3dVector(list(data.values())[0])
            else:
                self._pcd = o3d.io.read_point_cloud(str(self.path))
        return self._pcd
    
    @pcd.setter
    def pcd(self, value: o3d.geometry.PointCloud) -> None:
        assert isinstance(value, o3d.geometry.PointCloud)
        self._pcd = value
    
    @property
    def arr(self) -> np.ndarray:
        self._arr = np.asfarray(self.pcd.points)
        return self._arr
    
    @arr.setter
    def arr(self, value: np.ndarray) -> None:
        assert isinstance(value, np.ndarray)
        self._arr = value
    
    @property
    def features(self) -> Union[torch.Tensor, None]:
        return self._feat
    
    @features.setter
    def features(self, value: torch.Tensor) -> None:
        assert torch.is_tensor(value)
        self._feat = value
        
    @property
    def index(self) -> int:
        return int(str(self.path.stem).replace("cloud_bin_", ""))
    
    def rand_downsample(self, size: int) -> None:
        """
        Random downsample

        Args:
            size: The desired size of the cloud
        """
        self._arr = self._arr[np.random.permutation(self._arr.shape[0])[:size]]
        self._pcd.points = o3d.utility.Vector3dVector(self._arr)
        
    def prob_subsample(self, size: int, overlap_score: torch.Tensor, saliency_score: torch.Tensor) -> 'Cloud':
        """
        Probabilistic downsample
        
        Selects the points of the cloud that are most important for registration, based on the given overlap and
        saliency scores

        Args:
            size: How many samples we need
            overlap_score: The overlap scores for each point in the original cloud
            saliency_score: The saliency scores for each point in the original cloud

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
    
    def paint(self, rgb: Union[List, np.ndarray], cmap: str = "RdBu", computeNormals: bool = False) -> None:
        """
        Paints the point cloud
        
        If a simple 3 value list is given then we assume its a uniform color for all points in the point cloud
        If an array is given we expect a (N, 3) array with RGB values for each point in the pointcloud
        
        I would've like to use match case here but we're stuck with python 3.8

        Args:
            rgb: Either a single RGB color or one RGB color for each point in the point cloud
            cmap: Which matplotlib colormap to use when rgb is an array
            computeNormals: Wether to compute the point cloud normals or not
        """
        if isinstance(rgb, list):
            if any(value > 1 for value in rgb): # Open3D excpects colors in the [0, 1] range
                rgb = [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255]
            self.pcd.paint_uniform_color(np.array(rgb))
            
        elif isinstance(rgb, np.ndarray):
            assert len(rgb) == len(self), f"Colors array ({rgb.shape}) must match shape of point cloud {self.arr.shape}"
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) # min-max normalization
            cmap = mpl.colormaps[cmap]
            self.pcd.colors = o3d.utility.Vector3dVector(cmap(rgb)[:, :3])
            
        if computeNormals:
            self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
