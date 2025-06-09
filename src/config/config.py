from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import Field, field_serializer
from typing import Dict, Any, Optional, Self
import yaml
from pathlib import Path
import msgpack


class SerializableSettings(BaseSettings):
    def save(self, path: Path) -> None:
        with (path / "config.msgpack").open("wb") as f:
            msgpack.dump(self.model_dump(), f, use_bin_type=True)

    @classmethod
    def load(cls, path: Path) -> "SerializableSettings":
        with (path / "config.msgpack").open("rb") as f:
            data = msgpack.load(f, raw=False)
        return cls(**data)


class AvailableModules(str, Enum):
    IAE = "IAE"
    PREDATOR = "PREDATOR"
    PREDATOR_PRE_TRAINED = "PREDATOR"


class Model(SerializableSettings):
    class Encoder(SerializableSettings):
        N_HIDDEN_LAYERS: int
        GRID_RES: int
        FREEZE: bool = False

    class Decoder(SerializableSettings):
        HIDDEN_SIZE: int
        N_BLOCKS: int

    MODULE: AvailableModules
    PADDING: float
    ENCODER: Encoder
    LATENT_DIM: int
    DECODER: Decoder


class Transforms(SerializableSettings):
    TRAIN: Optional[Dict[str, Dict[str, Any]]] = {}
    VAL: Optional[Dict[str, Dict[str, Any]]] = {}


class AvailableOptimizers(str, Enum):
    ADAM = "adam"
    SGD = "sgd"


class LearningRate(SerializableSettings):
    OPTIMIZER: AvailableOptimizers
    VALUE: float
    WEIGHT_DECAY: Optional[float] = None
    MOMENTUM: Optional[float] = None


class AvailableTrainingModes(str, Enum):
    NEW = "new"
    LATEST = "latest"


class Trainer(SerializableSettings):
    MODE: AvailableTrainingModes
    BACKUP_DIR: Path = Path("./src/LIM/training/backups/")
    BATCH_SIZE: int
    LEARNING_RATE: LearningRate
    EPOCHS: int
    ACCUM_STEPS: int = Field(default=1, ge=1)
    VALIDATION_PERIOD: int
    BACKUP_PERIOD: int
    VALIDATION_SPLIT: float
    MULTIPROCESSING: bool
    POINTCLOUD_TF: Optional[Transforms] = None
    IMPLICIT_GRID_TF: Optional[Transforms] = None

    @field_serializer("BACKUP_DIR")
    def serialize_backup_dir(self, backup_dir: Path, _info) -> str:
        return str(self.BACKUP_DIR)


class Tester(SerializableSettings):
    class Ransac(SerializableSettings):
        MAX_ITERATIONS: int
        MAX_VALIDATION_STEPS: int
        SIMILARITY_THRESHOLD: float
        DISTANCE_THRESHOLD: float

    BATCH_SIZE: int
    DOWNSAMPLE_SIZE: int
    RANSAC: Ransac


class Settings(SerializableSettings):
    MODEL: Model
    TRAINER: Trainer
    TESTER: Tester
    DEVICE: str

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


settings = None  # Set from within main.py depending on which model we're training
