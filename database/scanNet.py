from torch.utils.data import Dataset
from database.cloud import Cloud
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from zipfile import ZipFile
import open3d as o3d

class ScanNet(Dataset):
    """
    Class representing the ScanNet dataset
    
    The hierarchy of the files goes like this:
        * room:
            * scene:
                * points_iou:
                    * points_iou_xx.npz
                * pointcloud:
                    * pointcloud_xx.npz

    Here the actual pointclouds are stored in the ['points'] field inside the pointcloud_xx.npz files
    
    """
    
    dir: Path
    clouds: List[Tuple[Cloud, Cloud]]
    
    def __init__(self) -> None:
        self.dir = Path("database/raw_data/scannet.zip")
        self.clouds = []
        
        with ZipFile(self.dir, 'r') as contents:
            for filename in contents.namelist():
                path = Path(filename)
                if ("pointcloud" in path.parts) and (path.suffix.lower() in ['.npz', '.npy']):
                    xx = path.stem[-2:] # xx is explained in the class definition
                    iou_path = path.parent.parent / f"points_iou/points_iou_{xx}.npz"
                    self.clouds.append((Cloud(path), Cloud(iou_path)))
        
        print(f"Loaded ScanNet with {len(self)} point clouds")
        
    def __len__(self) -> int:
        return len(self.clouds)
        
    def __getitem__(self, idx: int) -> Tuple[Cloud, Cloud]:
        """
        Extracts from the scannet zipfile the contents of just the pointcloud associated with idx.
        
        In the IAE's processed version of scannet they include the following data for each pointcloud:
            * df_value:     The computed value for the Distance Function (DF)
            * occupancies:  ???
            * points:       The points representing the discretized space, each point has a DF value associated
        """
        with ZipFile(self.dir, 'r') as contents:
            pcd_path = str(self.clouds[idx][0].path)
            pcd = o3d.geometry.PointCloud()
            with contents.open(pcd_path) as file:
                pcd.points = o3d.utility.Vector3dVector(np.load(file)['points'])
                self.clouds[idx][0].pcd = pcd
            
            iou_path = str(self.clouds[idx][1].path)
            dfpcd = o3d.geometry.PointCloud()
            with contents.open(iou_path) as file:
                npz = np.load(file)
                # occupancies = npz['occupancies']
                dfpcd.points = o3d.utility.Vector3dVector(npz['points'])
                self.clouds[idx][1].pcd = dfpcd
                self.clouds[idx][1].paint(npz['df_value'], cmap="RdBu", computeNormals=False)
                
        return self.clouds[idx]