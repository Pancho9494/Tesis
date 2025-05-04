from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Dict, Any, Optional, List
import yaml


def yaml_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    with open("./config/config.yaml", "r") as f:
        return yaml.safe_load(f)


class Model(BaseSettings):
    class Encoder(BaseSettings):
        N_HIDDEN_LAYERS: int
        GRID_RES: int

    class Decoder(BaseSettings):
        HIDDEN_SIZE: int
        N_BLOCKS: int

    PADDING: float
    ENCODER: Encoder
    LATENT_DIM: int
    DECODER: Decoder


class Transforms(BaseSettings):
    TRAIN: Optional[Dict[str, Dict[str, Any]]] = {}
    VAL: Optional[Dict[str, Dict[str, Any]]] = {}


class Trainer(BaseSettings):
    BATCH_SIZE: int
    LEARNING_RATE: float
    WEIGHT_DECAY: Optional[float] = None
    MOMENTUM: Optional[float] = None
    EPOCHS: int
    ACCUM_STEPS: int = Field(default=1, ge=1)
    VALIDATION_PERIOD: int
    BACKUP_PERIOD: int
    VALIDATION_SPLIT: float
    MULTIPROCESSING: bool
    POINTCLOUD_TF: Optional[Transforms] = None
    IMPLICIT_GRID_TF: Optional[Transforms] = None


class Tester(BaseSettings):
    class Ransac(BaseSettings):
        MAX_ITERATIONS: int
        MAX_VALIDATION_STEPS: int
        SIMILARITY_THRESHOLD: float
        DISTANCE_THRESHOLD: float

    BATCH_SIZE: int
    DOWNSAMPLE_SIZE: int
    RANSAC: Ransac


class Settings(BaseSettings):
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
