import json
import random
from pathlib import Path

import cv2
import numpy as np

# 預設 config (.yaml 無指定則使用此預設)
DEFAULT_TARGET_LABEL = "cell"
DEFAULT_PREVIEW_SAMPLE_RATE = 0.2
DEFAULT_PREVIEW_SEED = 77

def setup_mask_output_dir(config):
    path = config["path"]
    masks_output = path["masks_output"]
    masks_check = path["masks_check"]

    for folder in (masks_output, masks_check):
        folder.mkdir(parents=True, exist_ok=True)

    return masks_output, masks_check