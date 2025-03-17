from abc import ABC, abstractmethod
from pathlib import Path
from config import settings
import torch
import inspect


class Model(ABC, torch.nn.Module):
    __device: torch.device = torch.device(settings.DEVICE)

    def __init__(self, *args, **kwargs) -> None:
        super(Model, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> any: ...

    def load(self, suffix: str = "") -> None:
        """
        Load model weights from a .tar file in the 'weights' directory.

        Args:
            suffix (str, optional): Suffix to append to the filename. Defaults to "".

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If "model_state_dict" is missing in the file.
        """

        self.load_state_dict(
            torch.load(
                Path(inspect.getfile(self.__class__)).parent / "weights" / f"{self.__class__.__name__}_{suffix}.tar",
                weights_only=True,
            )["model_state_dict"]
        )

    def save(self, suffix: str = "") -> None:
        """
        Save model weights to a .tar file in a 'weights' directory located
        next to the `.py` file where the model is implemented.

        Args:
            suffix (str, optional): A string suffix to append to the filename of the saved
                weights file. Defaults to an empty string.
        """

        torch.save(
            {
                "model_state_dict": self.state_dict(),
            },
            Path(inspect.getfile(self.__class__)).parent / "weights" / f"{self.__class__.__name__}_{suffix}.tar",
        )
