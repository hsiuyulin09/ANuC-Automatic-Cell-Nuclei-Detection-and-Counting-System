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

# python main.py init
# python main.py preprocessing
# python main.py mask
# python main.py package
# python main.py train run
# python main.py anuc init
# python main.py anuc predict

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
    
    if args.command == "train" and args.training_command == "run":
        return load_config(DEFAULT_TRAINING_CONFIG)
    
    if args.command == "anuc" and args.aunc_commands == "init":
        return load_config(DEFAULT_PREDICTION_CONFIG)
    
    raise ValueError(f"unsupported command for config loading. {args.command}") # user 指令無法識別則跳脫
    
def load_anuc_predict_config(args): # anuc predict
    load_predict_config = args.config or DEFAULT_PREDICTION_CONFIG # 這裡的 or 會優先讀取左邊的參數

    predict_config = load_config(load_predict_config)
    postprocessing_config = load_config(DEFAULT_POSTPROCESSING_CONFIG)

    return predict_config, postprocessing_config


def run_init(args):
    config = load_command_config(args)
    setup_preprocessing_input_dir(config)

def run_processing(args):
    config = load_command_config(args)
    process_images(config)

def run_mask(args):
    from src.labelme_masks import convert_labelme_json_to_masks

    config = load_command_config(args)
    convert_labelme_json_to_masks(config)

def run_package(args):
    from src.dataset_h5 import create_h5_dataset

    config = load_command_config(args)
    create_h5_dataset(config)

def run_train(args):
    from src.train import train_model

    config = load_command_config(args)
    train_model(config)

def run_anuc_init(args):
    from src.predict import setup_prediction_output_dir

    config = load_command_config(args)
    setup_prediction_output_dir(config)

def run_anuc_predict(args):
    from src.orchestrator import run_prediction_pipeline

    predict_config, postprocessing_config = load_anuc_predict_config(args)
    run_prediction_pipeline(predict_config, postprocessing_config)


def dispatch(args):
    if args.command == "init":
        run_init(args)
        return
    
    if args.command == "preprocessing":
        run_processing(args)
        return
    
    if args.command == "mask":
        run_mask(args)
        return
    
    if args.command == "package":
        run_package(args)
        return
    
    if args.command == "train" and args.training_command == "run":
        run_train(args)
        return
    
    if args.command == "anuc" and args.aunc_commands == "init":
        run_anuc_init(args)
        return
    
    if args.command == "anuc" and args.anuc_commands == "predict":
        run_anuc_predict(args)
        return
    
    raise ValueError(f"unsupported command. {args}")

def main():
    args = parse_args()

    try:
        dispatch(args)
    except FileNotFoundError as error: # 檔案或路徑不存在
        print(f"error: {error}")
        sys.exit(1)
    except KeyError as error: # config YAML 結構不完整
        print(f"missing config key: {error}")
        sys.exit(1)
    except ValueError as error: # 參數值不合理
        print(f"error: {error}")
        sys.exit(1)

if __name__ == "__main__":
    main()
