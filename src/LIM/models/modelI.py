import os
from abc import ABC, abstractmethod
from copy import copy
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Qt5Agg")
import matplotlib

matplotlib.use(os.environ["MPLBACKEND"], force=True)
import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy as msgpk_np
import numpy as np
import open3d as o3d
import torch
import umap
from sklearn.manifold import TSNE

from LIM.data.structures import Pair, PCloud
from LIM.log import log
from LIM.training.threading import backup_executor

msgpk_np.patch()  # Numpy compatibility


class Model(ABC, torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Model, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> any: ...

    def load(self, run: str | Path | os.PathLike = "", suffix: str = "") -> None:
        """
        Load model weights from a .msgpack file in the training/backups dir.

        Args:
            suffix (str, optional): Suffix to append to the filename. Defaults to "".

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If "model_state_dict" is missing in the file.
        """
        path = Path(f"{run}/model_{suffix}.msgpack")
        if not path.exists():
            log.error(f"Couldn't find Model backup at {path}")
            raise FileNotFoundError

        log.info(f"Model loading backup from {path}")

        with path.open("rb") as f:
            data = msgpack.unpackb(f.read(), raw=False)

        if "model_state_dict" not in data:
            log.error(f"Missing 'model_state_dict' in loaded backup {path}")
            raise KeyError

        self.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in data["model_state_dict"].items()})

    def save(self, run: str = "", suffix: str = "") -> None:
        """
        Save model weights to a .msgpack file in the training/backups dir.

        Args:
            suffix (str, optional): A string suffix to append to the filename of the saved
                weights file. Defaults to an empty string.
        """
        path = Path(f"{run}/model_{suffix}.msgpack")
        log.info(f"Model saving backup to {path}")

        with path.open("wb") as f:
            data = msgpack.packb(
                {"model_state_dict": {k: np.copy(v.cpu().numpy()) for k, v in self.state_dict().items()}},
                use_bin_type=True,
            )
            f.write(data)

    def save_async(self, run: str | Path | os.PathLike = "", suffix: str = "") -> None:
        backup_executor.submit(self.save, run, suffix)

    def _show_latent_space(self, pair: Pair, reducer: TSNE | umap.UMAP) -> None:
        log.info(f"Generating latent space visualization for {self.__class__.__name__}")

        log.info(f"{pair=}")
        self.eval()
        with torch.no_grad():
            features = np.vstack([pair.source.features.cpu().numpy(), pair.target.features.cpu().numpy()])
            log.info(f"{features.shape=}")

            source_points = pair.source.points.cpu().numpy()
            target_points = pair.target.points.cpu().numpy()
            log.info(f"{source_points.shape=}, {target_points.shape=}")

            aligned_source = copy(pair.source)
            aligned_target = copy(pair.target)
            aligned_source.pcd = aligned_source.pcd.to_legacy().transform(pair.GT_tf_matrix)
            aligned_target.pcd = aligned_target.pcd.to_legacy()
            points = np.vstack([aligned_source.arr, aligned_target.arr])
            log.info(f"{points.shape=}")

            y_coords = points[:, 1]
            colors = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())

            source_colors = colors[: len(source_points)]
            target_colors = colors[len(source_points) :]

        aligned_source.pcd.colors = o3d.utility.Vector3dVector(plt.cm.viridis(source_colors)[:, :3])
        aligned_target.pcd.colors = o3d.utility.Vector3dVector(plt.cm.viridis(target_colors)[:, :3])
        o3d.visualization.draw_geometries([aligned_source.pcd, aligned_target.pcd], window_name="Bunny by y-axis")

        log.info(f"Running dimensionality reduction with {reducer}")
        embedding = reducer.fit_transform(features)
        log.info("Successfully reduced dimensionality")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=colors,
            cmap="viridis",
            s=5,
        )
        ax.set_title(f"{reducer} Latent space visualization")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        fig.colorbar(scatter, label="Normalized Heigth")
        plt.show()

    def show_umap(self, pair: Pair) -> None:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
        self._show_latent_space(pair, reducer)

    def show_tsne(self, pair: Pair) -> None:
        reducer = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42)
        self._show_latent_space(pair, reducer)
