import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import BCEDiceLoss, H5Dataset, UNet, set_device

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 60
DEFAULT_MODEL_SEED = 77
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_THRESHOLD = 0.5

def train_model(config):
    paths = config["paths"]
    train_config = config.get("training") or {}

    h5_path = Path(paths["dataset_output"])
    model_output = Path(paths["model_output"])

    batch_size = train_config.get("batch_size")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE

    epochs = train_config.get("epochs")
    if epochs is None:
        epochs = DEFAULT_EPOCHS

    seed = train_config.get("seed")
    if seed is None:
        seed = DEFAULT_MODEL_SEED

    learning_rate = train_config.get("learning_rate")
    if learning_rate is None:
        learning_rate = DEFAULT_LEARNING_RATE

    threshold = train_config.get("threshold")
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    if not h5_path.exists():
        raise FileNotFoundError(f"h5 dataset not exist: {h5_path}")

    model_output.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 固定weight initiation的random
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)