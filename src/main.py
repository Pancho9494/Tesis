from rich.traceback import install

install(show_locals=False)

import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

from LIM.models.IAE.iae import IAE
from LIM.models.trainer import Trainer
from LIM.data.datasets.scanNet import ScanNet
import open3d as o3d

import matplotlib as mpl
import numpy as np


def trainIAE():
    dataset = ScanNet()
    print(torch.cuda.is_available())
    for it, (cloud, implicit) in enumerate(dataset):
        cloud.paint([255, 255, 0], computeNormals=True)
        print(cloud.pcd.point.positions.device)

        o3d.visualization.draw_geometries([cloud.pcd])
    # trainer = Trainer(IAE(), ScanNet())
    # trainer.train(plot=False)


if __name__ == "__main__":
    trainIAE()
