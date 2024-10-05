import numpy as np
import open3d as o3d
from pathlib import Path
from zipfile import ZipFile
from typing import List, Tuple
from database.cloud import Cloud
from torch.utils.data import Dataset

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
    clouds: List[Tuple[Cloud, Cloud]] # Tuple[The actual point cloud, The Distance Function point cloud]
    
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
                print(npz.files)
                # occupancies = npz['occupancies']
                dfpcd.points = o3d.utility.Vector3dVector(npz['points'])
                self.clouds[idx][1].pcd = dfpcd
                self.clouds[idx][1].paint(npz['df_value'], cmap="YlGnBu", computeNormals=False)
                
        # IAE applies a SubsamplePintcoud transformation, which just selects 2048 points from the dfpcd point cloud
        """
        I'm trying to figure out how do they get the data['points', 'points.df', 'inputs'] dictionary
        they use in IAE.src.dfnet.training.Trainer.compute_loss()
        
        They declare various Field classes that apply different transformations to the files in order to generate the
        dictionary, so far I have figured out that:
            * IAE.src.data.core.Shapes3dDataset is the core Dataset class
            * in its __getitem__ method they generate the data dictionary by applying these Field classes
            * For the 'points' field:
                o They use the PointsField() class 
                o Used for the points randomly sampled in the bounding volume of the 3D shape, 
                  i.e. the cube we call dfpcd
                o If there's more than one file in the "points_iou" folder, they randomly choose one
                o The dictionary at this stage is:
                    data = {
                        None: np.load(iou_path)['points'],
                        'df': np.load(iou_path)['df_value']
                    }
                o It applies a SubsamplePoints() transform, which selects 2048 from both the points and the df_values
                o So the dictionary is:
                    data = {
                        None: subsampled(np.load(iou_path)['points']),
                        'df': subsampled(np.load(iou_path)['df_value'])
                    }
                
            * For the 'input' field:
                o They use the PartialPointCloudField() class
                o Used for the points randomly sampled on the mesh and a bounding box with ransdom size is applied,
                  i.e. these are the actual points from npz['points'] but I have no idea what do they mean by the
                  bounding box
                o If there's more than one file in the "points_iou" folder, they randomly choose one
                o They load the npz['points'] data (what we call pcd) to the "pointcould_dict" variable, which then they
                  fucking ignore and just load the actual array to a "points" variable.
                  
                  The PartialPointCloudField does different things to the points variable depending on the
                  "partial_type" defined in the config .yaml file, that in the case of scannet, is "centerz_random" so
                  they crop the pointcloud in the x and y axes in a random proportion, at most half
                o The dictionary at this stage is:
                    data = {
                        None: cropped(np.load(pcd_path)['points'])
                    }
                o Then it applies SubsamplePointCloud(), which just randomly chooses 2048 points, so the dict is:
                    data = {
                        None: subsampled(cropped(np.load(pcd_path)['points']))
                    }
                o And then it applies PointCloudNoise(), which applies gaussian noise with a standard deviation of
                0.005, so the dict is:
                    data = {
                        None: noise(subsampled(cropped(np.load(pcd_path)['points'])))
                    }
            
            * These dictionaries later get parsed in the format "field_name.key", and the None keys get the field name
              so we have:
                data = {
                    points = subsampled(np.load(iou_path)['points']),
                    points.df = subsampled(np.load(iou_path)['df_value']),
                    inputs = noise(subsampled(cropped(np.load(pcd_path)['points'])))
                }
        
        Now that we know wtf was going on with the data dictionary, we know that the encoder receives as input the
        noised + subsampled + cropped point cloud, and then that output is passed to the decoder, along with the
        subsampled cube points, and then we compute the loss with the encoder output and the subsampled df_values
        """
                
        return self.clouds[idx]