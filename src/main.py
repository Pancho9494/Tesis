from rich.traceback import install

install(show_locals=False)

from LIM.models.IAE.iae import IAE
from LIM.models.trainer import Trainer
from LIM.data.datasets.scanNet import ScanNet
import open3d as o3d


def show():
    dataset = ScanNet()
    for it, (cloud, implicit) in enumerate(dataset):
        cloud.paint([255, 255, 0], computeNormals=True)
        implicit.paint(implicit.features, cmap="YlGnBu", computeNormals=False)
        o3d.visualization.draw_geometries(
            [
                cloud.pcd.to_legacy(),
                # implicit.pcd.to_legacy(),
            ]
        )


def trainIAE():
    trainer = Trainer(IAE(), ScanNet())
    trainer.train(plot=False)


if __name__ == "__main__":
    trainIAE()
    # show()
