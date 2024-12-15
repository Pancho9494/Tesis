import asyncio
import sys
from from_root import from_root

sys.path.append(f"{from_root()}/src/submodules/IAE")
from LIM.models.IAE.DGCNNEncoder import DGCNN
from LIM.models.IAE.iae import IAE
from LIM.models.trainer import Trainer
from LIM.data.datasets.scanNet import ScanNet

import open3d as o3d
from rich.traceback import install
from config import settings


install(show_locals=False)


def show():
    dataset = ScanNet()
    for it, (cloud, implicit) in enumerate(dataset):
        cloud.paint([255, 255, 0], computeNormals=True)
        implicit.paint(implicit.features, cmap="YlGnBu", computeNormals=False)
        o3d.visualization.draw_geometries(
            [
                cloud.pcd.to_legacy(),
                implicit.pcd.to_legacy(),
            ]
        )


def cleanDataset():
    dataset = ScanNet()
    asyncio.run(dataset.clean_bad_files())


def trainIAE():
    model = IAE(
        DGCNN(
            knn=settings.MODEL.ENCODER.KNN,
            emb_dims=settings.MODEL.ENCODER.EMB_DIM,
            latent_dim=settings.MODEL.LATENT_DIM,
        )
    )
    dataset = ScanNet()
    trainer = Trainer("IAE_Training", model, dataset)
    # trainer.load_model("./weights/IAE_Training.tar")
    trainer.train()


if __name__ == "__main__":
    # cleanDataset()
    trainIAE()
    # show()
