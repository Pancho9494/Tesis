from customModels.predator import Predator
from database.threeDLoMatch import ThreeDLoMatch
import open3d as o3d
import numpy as np
import copy

# pytorch with rocm won't run without this
from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("PYTORCH_ROCM_ARCH", "gfx1031")

def main():
    model = Predator("indoor")
    data = ThreeDLoMatch()
    
    for sample in data:
        transform = model(sample)
        
        # # TODO: now we need a class that handles the showing of results, and computes metrics
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(sample.src.arr)
        src_pcd.paint_uniform_color(np.array([1, 0.706, 0]))
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(sample.target.arr)
        target_pcd.paint_uniform_color(np.array([0, 0.651, 0.929]))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        
        registration_pcd = copy.deepcopy(src_pcd)
        registration_pcd.transform(transform)
        
        def _rotate_view(vis):
            ctr = vis.get_view_control()
            ctr.rotate(2.0, 0.0)
            return False
        o3d.visualization.draw_geometries_with_animation_callback([target_pcd, registration_pcd], _rotate_view)
    
        
        
if __name__ == "__main__":
    main()