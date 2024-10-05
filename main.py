from customModels.predator import Predator
from database.threeDLoMatch import ThreeDLoMatch
from database.scanNet import ScanNet
from metrics.metrics import RegistrationRecall, RootMeanSquaredError
from typing import List
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from os import putenv # pytorch with rocm won't run without this
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("PYTORCH_ROCM_ARCH", "gfx1031")


def plot(model: str, overlaps: List[float], metrics: List[float]) -> None:
    fig, ax = plt.subplots()
    ax.plot(overlaps, metrics, label=model)
    ax.legend()
    plt.show()

def main():
    model = Predator("indoor")
    data = ThreeDLoMatch()
    RR = RegistrationRecall()
    RMSE = RootMeanSquaredError()
    
    subset =  torch.utils.data.Subset(data, list(range(0, 5)))
    
    overlaps = []
    recalls = []
    for pair in subset:
        pair.prediction = model(pair)
        pair.show(num_steps=100)
        recalls.append(RMSE(pair))
        overlaps.append(pair.overlap) # TODO: maybe overlaps should be saved in the Pairs class
    
    plot(repr(model), overlaps, recalls)
    
    
from customModels.IAE.iae import IAE
import open3d as o3d

def testIAE():
    # data = ThreeDLoMatch()
    # for pair in data:
    #     iae(pair.src)
    model = IAE(IAE.Implicit.UDF)
    data = ScanNet()
    for pcd, df_pcd in data:
        pcd.paint([255, 221, 0], computeNormals=True)
        o3d.visualization.draw_geometries([
            pcd.pcd,
            df_pcd.pcd
        ])
        
        model.encoder(pcd)
        

    
if __name__ == "__main__":
    testIAE()