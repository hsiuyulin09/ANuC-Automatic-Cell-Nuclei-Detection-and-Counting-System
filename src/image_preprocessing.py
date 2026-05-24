from pathlib import Path
import cv2
import sys

def setup_preprocessing_input_dir(config):
    stage1_config = config["stage1"]
    paths = stage1_config["paths"]
    folder = paths["preprocessing_input"]

    Path(folder).mkdir(parents=True, exist_ok=True)
        #Path(f)將抓到的字串轉成一個路徑物件, .mkdir用前面轉換的路徑建立一個資料夾
        #parents=True偵測未建立的父目錄，未建立就自動建立 exist_ok=True偵測已建立的資料夾就跳過

    print(f"{folder} has been created")

def setup_preprocessing_output_dir(config):
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

def enhance_image(img, clahe_config, bilateral_config):
    denoised_img = cv2.bilateralFilter(img, bilateral_config["d"], bilateral_config["sigma_color"], bilateral_config["sigma_space"])
        # d 代表以目標像素為中心考慮的區域直徑
        # sigmaColor 決定多大的差異會被視為邊緣
        # sigmaSpace 影響力隨距離衰減的速度

    clahe = cv2.createCLAHE(clipLimit=clahe_config["clip_limit"], tileGridSize=tuple(clahe_config["tile_grid_size"]))
        # yaml 用 list 寫但 createCLAHE 要讀取要轉成 tuple
        # cv2.createCLAHE 用 OpenCV 建立 CLAHE 處理器
        # clipLimit 設定對比限制
        # tileGridSize 指定局部區塊大小

    enhanced_image = clahe.apply(denoised_img)

    return enhanced_image

def iter_tiles(image, tile_size, overlap): # sliding window
    h, w = image.shape
    stride = tile_size - overlap

    if overlap >= tile_size:
        print(f"error: overlap size cannot equal or larger than tile size")
        sys.exit(1)

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            # image[y:y+tile_size, x:x+tile_size]
            # y 控制 row 範圍，x 控制 column 範圍
            tile = image[y:y + tile_size, x:x + tile_size]
            yield y, x, tile

def processing_images(config):
    stage1_config = config["stage1"]

    paths = stage1_config["paths"]
    preprocessing_input = Path(paths["preprocessing_input"])
    preprocessing_output = Path(paths["preprocessing_output"])
    preprocessing_check = Path(paths["preprocessing_check"])

    preprocessing = stage1_config["preprocessing"]
    tile_size = preprocessing["tile_size"]
    overlap = preprocessing["overlap"]
    image_extensions = preprocessing["image_extensions"]

    clahe_config = preprocessing["clahe"]

    bilateral_config = preprocessing["bilateral_filter"]

    if not preprocessing_input.exists():
        print(f"error: input directory ({preprocessing_input}) does not exist")
        print("run `python main.py train init` first")
        sys.exit(1)
    
    setup_preprocessing_output_dir(config)

    image_files = list_input_images(preprocessing_input, image_extensions)

    if not image_files:
        print(f"warning: no input images found in {preprocessing_input}")
        return
    
    print(f"total image: {len(image_files)}")

    for image_path in image_files:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None: # cv2.imread 讀取失敗處理
            print(f"warning: failed to read image: {image_path}")
            continue

        enhanced_image = enhance_image(img, clahe_config, bilateral_config)

        cv2.imwrite(str(preprocessing_check/f"enhanced_{image_path.name}"), enhanced_image) # 存給 debug (check) 用的

        count = 0

        for y, x, tile in iter_tiles(enhanced_image, tile_size, overlap):
            filename = f"{image_path.stem}_y{y}_x{x}.png"
            if not cv2.imwrite(str(preprocessing_output / filename), tile):
                print(f"warning: failed to write tile: {filename}")
            count+=1

        print(f"{image_path.name}: {count} tiles")