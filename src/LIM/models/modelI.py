from abc import ABC, abstractmethod
from pathlib import Path
import torch
import config.config as config
import LIM.log as log
import msgpack
import msgpack_numpy as msgpk_np
import numpy as np
from LIM.training.threading import backup_executor

msgpk_np.patch()  # Numpy compatibility


class Model(ABC, torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Model, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> any: ...

    def load(self, run: str = "", suffix: str = "") -> None:
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

    def save_async(self, run: str = "", suffix: str = "") -> None:
        backup_executor.submit(self.save, run, suffix)
