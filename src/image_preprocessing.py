import cv2
from pathlib import Path

def setup_train_folders(config):
    stage1_config = config["stage1"]
    paths = stage1_config["paths"]
    folder = paths["preprocessing_input"]

    Path(folder).mkdir(parents=True, exist_ok=True)
        #Path(f)將抓到的字串轉成一個路徑物件, .mkdir用前面轉換的路徑建立一個資料夾
        #parents=True偵測未建立的父目錄，未建立就自動建立 exist_ok=True偵測已建立的資料夾就跳過

    print(f"{folder} has been created")

def setup_preprocessing_output_folders(config):
    stage1_config = config["stage1"]
    paths = stage1_config["paths"]
    folders = [paths["preprocessing_output"], paths["preprocessing_check"]]

    for folder in folders:

        Path(folder).mkdir(parents=True, exist_ok=True)

    print(f"Folder has been created: {folders}")

def list_input_images(input_folder, image_extensions):
    input_path = Path(input_folder)

    image_files = [
        path for path in input_path.iterdir()
        # Path(path).suffix.lower() 取得副檔名並轉小寫
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    return image_files