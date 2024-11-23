from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any
import yaml


def yaml_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    with open("./config/config.yaml", "r") as f:
        return yaml.safe_load(f)


class Encoder(BaseSettings):
    KNN: int
    EMB_DIM: int
    PADDING: float


class Decoder(BaseSettings):
    HIDDEN_SIZE: int
    N_BLOCKS: int
    PADDING: float


class Model(BaseSettings):
    ENCODER: Encoder
    LATENT_DIM: int
    DECODER: Decoder


class Trainer(BaseSettings):
    BATCH_SIZE: int
    LEARNING_RATE: float
    EPOCHS: int
    ACCUM_STEPS: int
    VALIDATION_PERIOD: int
    BACKUP_PERIOD: int
    VALIDATION_SPLIT: float
    MULTIPROCESSING: bool


class Settings(BaseSettings):
    MODEL: Model
    TRAINER: Trainer

    model_config = SettingsConfigDict(env_file="config.yaml", env_file_encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


settings = Settings.from_yaml("./src/config/config.yaml")
