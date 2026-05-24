import argparse
import sys
from pathlib import Path
from src.config import load_config
from src.image_preprocessing import process_images, setup_preprocessing_input_dir

# config 預設路徑
DEFAULT_PREPROCESSING_CONFIG = "configs/preprocessing_config.yaml"

def parse_args(): 
        # 解析 CLI 參數
        # 接收使用者輸入的 command
        # 支援 --config 覆蓋預設 YAML
        # 建立目前與未來預計支援的 command 結構
    parser = argparse.ArgumentParser(description="ANuC Automatic Cell Nuclei Detection and Counting System CLI control system")
        # description 即為 CLI 程式的說明文字 (python main.py --help)

    parser.add_argument(
        "--config",
        default=None,
        help="path to configS (YAML), if omitted, command-specific default config is used"
        ) 
        # parser.add_argument 建立的參數需要使用者的輸入, 使用者輸入類型取決於提取參數的程式需求什麼
        # 建立可選參數 --config, 讀取 configs/ # 可選參數代表 user 可填可不填
        # user 指定路徑優先, 無指定就用預設路徑

    subparser = parser.add_subparsers(dest="command", required=True)
        # 在 parser 主解析器下建立指令容器, 命名為"command"
        # equired=True 執行時 user 必填項目

    subparser.add_parser("init", help="create user-input folders required before preprocessing")
    subparser.add_parser("preprocessing", help="run image preprocessing and generate tiles for Labelme")