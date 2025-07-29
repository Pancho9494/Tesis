import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Self

import msgpack
import yaml
from pydantic import Field, computed_field, field_serializer, model_validator
from pydantic_settings import BaseSettings


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
    TOY_PREDATOR = "TOY PREDATOR"
    TOY_IAE = "TOY IAE"


class Model(SerializableSettings):
    """
    TODO: We should really make some subclasses to properly define which model needs what
    """

    class Encoder(SerializableSettings):
        N_HIDDEN_LAYERS: int  # Must be the same in PREDATOR and IAE
        GRID_RES: int | None = None  # IAE
        FREEZE: int = 0  # IAE
        PRE_TRAINED: bool = False  # PREDATOR
        PRE_TRAIN_DATE: str | None = None  # YYYYMMDD_HHMMSS

    class Decoder(SerializableSettings):
        HIDDEN_SIZE: int | None = None  # IAE
        N_BLOCKS: int | None = None  # IAE

    MODULE: AvailableModules
    PADDING: float | None = None
    ENCODER: Encoder
    LATENT_DIM: int
    DECODER: Decoder


class Transforms(SerializableSettings):
    TRAIN: dict[str, dict[str, Any]] | None = Field(default_factory=dict)
    VAL: dict[str, dict[str, Any]] | None = Field(default_factory=dict)

    @property
    def TOY_TRAIN(self) -> dict[str, dict[str, Any]] | None:
        return self.TRAIN

    @property
    def TRAIN_TOY(self) -> dict[str, dict[str, Any]] | None:
        return self.TRAIN

    @property
    def TRAIN_LIBRARY(self) -> dict[str, dict[str, Any]] | None:
        return self.TRAIN

    @property
    def TOY_VAL(self) -> dict[str, dict[str, Any]] | None:
        return self.VAL

    @property
    def VAL_TOY(self) -> dict[str, dict[str, Any]] | None:
        return self.VAL

    @property
    def VAL_LIBRARY(self) -> dict[str, dict[str, Any]] | None:
        return self.VAL


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
    FIXED = "fixed"


class DistributedSettings(SerializableSettings):
    class Config:
        extra = "allow"

    BACKEND: str = "nccl"
    MASTER_ADDR: str = "localhost"
    MASETR_PORT: int = 12355

    @computed_field
    @property
    def WORLD_SIZE(self) -> int:
        return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 0

    @property
    def ON(self) -> bool:
        return self.WORLD_SIZE > 1

    @computed_field
    @property
    def RANK(self) -> int:
        return int(os.environ["RANK"]) if "RANK" in os.environ else 0

    @computed_field
    @property
    def LOCAL_RANK(self) -> int:
        return int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0


class Trainer(SerializableSettings):
    MODE: AvailableTrainingModes
    DATED: str | None = None  # Optional date to load from YYYYMMDD_HHMMSS
    SUBSET: str | None = None
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

    @model_validator(mode="after")
    def fixed_mode_needs_date(self) -> Self:
        if self.MODE == AvailableTrainingModes.FIXED and self.DATED is None:
            raise ValueError("Mode FIXED needs a value for DATED")
        elif self.MODE != AvailableTrainingModes.FIXED and self.DATED is not None:
            raise ValueError("DATED is only used for mode FIXED")
        return self


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
    DISTRIBUTED: DistributedSettings = Field(default_factory=DistributedSettings)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


settings = None  # Set from within main.py depending on which model we're training
