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

    tile_size = prediction_config.get("tile_size")
    if tile_size is None:
        tile_size = DEFAULT_TILE_SIZE

    overlap = prediction_config.get("overlap")
    if overlap is None:
        overlap = DEFAULT_OVERLAP

    image_extensions = prediction_config.get("image_extensions")
    if image_extensions is None:
        image_extensions = DEFAULT_IMAGE_EXTENSIONS

    if not input_path.exists():
        print(f"prediction input dir not exist. {input_path}")
        return
    
    if not model_path.exists():
        print(f"model weight file not exist. {model_path}")
        return

    file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_h5 = output_path / f"{file_name}.h5"

    device = set_device()
    model = UNet().to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
        # torch.load() 讀取 .pth, 把內容讀成 PyTorch 物件類似 dict
        # model.load_state_dict() 套入 model

    image_files = [
        img_path for img_path in sorted(input_path.iterdir())
        if img_path.is_file() and img_path.suffix().low() in image_extensions
    ]
        # .iterdir() 列出這個資料夾底下第一層的所有項目
        # .is_file() 檢查這個路徑是不是檔案, 回傳 boolean

    if not image_files:
        print("input image not exist")
        return
    
    print(f"total image: {len(image_files)}")

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    stride = tile_size - overlap
    weight_mask = generate_weight_mask(tile_size)

    with h5py.File(output_h5, "w") as f:
        for img_path in image_files:
            raw_img = cv2.imread(str(img_path)) # 解碼成np.array
            
            if raw_img is None:
                continue

            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)  # 為出原圖(給後處理heat map用), 此處才轉雙通道
            denoised = cv2.bilateralFilter(gray, 5, 75, 75)
            enhanced = clahe.apply(denoised)

            h, w = enhanced.shape

            # compute padding
            pad_h = (stride - (h - tile_size) % stride) % stride
            pad_w = (stride - (w - tile_size) % stride) % stride
            padded_img = cv2.copyMakeBorder(enhanced, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            # cv2.copyMakeBorder(image, 上邊增加數, 下邊增加數, 左邊增加數, 右邊增加數, 指定模式, 顏色)

            new_h, new_w = padded_img.shape
            prob_map = np.zeros((new_h, new_w), dtype=np.float32) # 建立 0 矩陣後面用於儲存標記
            count_map = np.zeros((new_h, new_w), dtype=np.float32)

            print(f"lading image. {img_path}")

            for y in range(0, new_h - tile_size + 1, stride):
                for x in range(0, new_w - tile_size + 1, stride):
                    tile = padded_img[y:y + tile_size, x:x + tile_size]

                    tile_t = torch.from_numpy(tile).detach().float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                    # torch.from_numpy()自numpy轉pytorch tensor #(Batch, Channel, Height, Width) = (1, 1, 256, 256)

                    with torch.no_grad():
                        output = model(tile_t)
                        prob = torch.sigmoid(output).cpu().squeeze().numpy()

                    prob_map[y:y + tile_size, x:x + tile_size] += prob * weight_mask
                    count_map[y:y + tile_size, x:x + tile_size] += weight_mask

            final_prob = np.divide(prob_map, count_map, where=count_map != 0, out=np.zeros_like(prob_map))
            final_prob = final_prob[:h, :w]  # [:h, :w]slice from 0 to h and w (delete padding)

            group = h5f.create_group(img_path.name)
            group.create_dataset("raw_image", data=raw_img)
            group.create_dataset("probability_map", data=final_prob)
            group.attrs["original_size"] = (h, w)  # 標記數據成分標籤

    print(f"{file_name} create complete")
