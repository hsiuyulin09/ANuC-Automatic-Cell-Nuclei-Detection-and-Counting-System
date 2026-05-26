import argparse
import sys
from pathlib import Path
from src.config import load_config
from src.image_preprocessing import process_images, setup_preprocessing_input_dir

# config 預設路徑
DEFAULT_PREPROCESSING_CONFIG = "configs/preprocessing_config.yaml"
DEFAULT_TRAINING_CONFIG = "configs/training_config.yaml"
DEFAULT_PREDICTION_CONFIG = "configs/prediction_config.yaml"
DEFAULT_POSTPROCESSING_CONFIG = "configs/postprocessing_config.yaml"

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
    subparser.add_parser("mask", help="convert Labelme JSON annotations to binary masks")
    subparser.add_parser("package", help="create H5 dataset from image tiles and masks")

    train_parser = subparser.add_parser("train", help="training command")
    train_subparser = train_parser.add_subparsers(dest="training_command", required=True)
    train_subparser.add_parser("run", help="train model from prepared H5 dataset")


    anuc_parser = subparser.add_parser("anuc", help="ANuC prediction commands")
    anuc_subparser = anuc_parser.add_subparsers(dest="aunc_commands", required=True)
    anuc_subparser.add_parser("init", help="prepare environment for ANuC prediction")
    anuc_subparser.add_parser("predict", help="run model prediction and postprocessing to get final result")

    args = parser.parse_args()

    return args

def load_command_config(args): # 讀取 CLI 對應 config(.yaml)
    
    if args.config is not None:
        return load_config(args.config)
    
    if args.command == "init":
        return load_config(DEFAULT_PREPROCESSING_CONFIG)
    
    if args.command == "preprocessing":
        return load_config(DEFAULT_PREPROCESSING_CONFIG)
    
    if args.command == "mask":
        return load_config(DEFAULT_TRAINING_CONFIG)
    
    if args.command == "package":
        return load_config(DEFAULT_TRAINING_CONFIG)
    
    if args.command == "train" and args.train_command == "run":
        return load_config(DEFAULT_TRAINING_CONFIG)
    
    if args.command == "anuc" and args.aunc_commands == "init":
        return load_config(DEFAULT_PREDICTION_CONFIG)