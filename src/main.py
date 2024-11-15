from rich.traceback import install

install(show_locals=False)

from LIM.models.IAE.iae import IAE
from LIM.models.trainer import Trainer
from LIM.data.datasets.scanNet import ScanNet
import open3d as o3d
import asyncio


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
    trainer = Trainer("IAE_Training", IAE(), ScanNet())
    # trainer.load_model("./weights/IAE_Training.tar")
    trainer.train()


if __name__ == "__main__":
    # cleanDataset()
    trainIAE()
    # show()
