import datetime
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
from src.model import UNet, generate_weight_mask, set_device


DEFAULT_TILE_SIZE = 256
DEFAULT_OVERLAP = 60
DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".tif", ".png"]

def setup_prediction_output_dir(config):
    paths = config["paths"]
    prediction_input = Path(paths["prediction_input"])

    prediction_input.mkdir(parents=True, exist_ok=True)
    
    print(f"initialization successful. please place images in: {prediction_input}")

def prediction_model(config):
    paths = config["paths"]
    prediction_config = config.get("prediction") or {}

    input_path = Path(paths["prediction_input"])
    output_path = Path(paths["prediction_results"])
    model_path = Path(paths["model_path"])
