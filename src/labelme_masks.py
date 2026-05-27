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
    paths = config["paths"]
    masks_output = Path(paths["masks_output"])
    masks_check = Path(paths["masks_check"])

    for folder in (masks_output, masks_check):
        folder.mkdir(parents=True, exist_ok=True)

    return masks_output, masks_check

def list_labelme_jsons(input_folder): # 取得 labelme json 檔的檔名清單 # 出一個 list
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError("lebalme input dir not exist")
    
    labelme_jsons_list = sorted(input_path.glob("*.json")) # sorted() 將找到的路徑排序並回傳 list

    return labelme_jsons_list

def convert_labelme_json_to_masks(config):
    paths = config["paths"]
    masks_config = config.get("masks") or {}
        # .get("a") 在 dict 中取得 a key 對應 value, .get("a", b) 無 a key 則使用 b 回傳

    input_folder = Path(paths["labelme_input"])

    target_label = masks_config.get("target_label")
    if target_label is None:
        target_label = DEFAULT_TARGET_LABEL

    preview_sample_rate = masks_config.get("preview_sample_rate")
    if preview_sample_rate is None:
        preview_sample_rate = DEFAULT_PREVIEW_SAMPLE_RATE

    preview_seed = masks_config.get("preview_seed")
    if preview_seed is None:
        preview_seed = DEFAULT_PREVIEW_SEED
    
    output_folder, preview_folder = setup_mask_output_dir(config)

    json_files = list_labelme_jsons(input_folder) # 取得檔案 list

    if not json_files:
        print("labelme files not exist")
        return
    
    print(f"total labelme files: {len(json_files)}")