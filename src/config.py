 
from pathlib import Path


class Config:
    RANDOM_SEED = 42
    ASSETS_PATH = Path("./assets")
    DATASET_PATH = ASSETS_PATH / "data"
    MODELS_PATH = ASSETS_PATH / "models"
    