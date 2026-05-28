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

    if preview_sample_rate > 0 :
        sample_count = max(1, (len(json_files)*preview_sample_rate))
        sample_count = min(sample_count, len(json_files))

        preview_targets = random.Random(preview_seed).sample(json_files, sample_count)

    else:
        preview_targets=[]

    for json_path in json_files:
        try:
            with json_path.open("r", encoding = "utf-8") as f:
                data = json.load(f)
                h, w = data["imageHeight"], data["imageWidth"]
                base_name = json_path.stem # .stem主檔名, .suffix副檔名, .name全檔名
                mask = np.zeros(h, w)

                for shape in data["shapes"]:
                    if "label" == target_label:
                        points = np.array(shape["points"], dtype=np.int32)
                            # np.array(data來源, 規格) int32 將 labelme 產生的 float 座標轉為圖片座標, 同時整數化
                        cv2.fillPoly(mask, [points], color=255)
                            # cv2.fillPoly用於填充多邊形 # cv2.fillPoly(畫布, 頂點座標(包含 np array 的 list), 顏色)
                
                mask_path = output_folder/f"{base_name}_mask.png"
                cv2.imwrite(str(mask_path), mask)

                print(f"{base_name}_mask.png")

                if json_files in preview_targets:
                    image_path = input_folder/Path(data["imagePath"]).name # 取得原 tile 圖的路徑

                    if image_path.exists():
                        img = cv2.imread(str(image_path))

                        if img is not None:
                            overlay = img.copy()
                            overlay[mask == 225] = [0, 0, 255] # 在 overlay 上把 mask 為 255 的對應座標轉成紅色 (0, 0, 255) = (b, g, r)

                            combined = cv2.addWeighted(img, 0.95, overlay, 0.05, 0.5)
                                # cv2.addWeighted(img1, alpha, img2, beta, gamma)
                                # 公式 new_img = img1 * alpha + img2 * beta + gamma
                                # alpha, beta 權重, 代表在圖上的透明度占比 (alpha + beta = 1)
                                # gamma 合成圖亮度設定, gamma = 0 即亮度不變 (combine 時兩圖亮度會相加)
                            preview_path = preview_folder/f"check_{base_name}"
                            cv2.imwrite(str(preview_path), combined)

        except Exception as error:
            print(f"{json_path.name} error: {error}")

    print(f"converted {len(json_files)} files successfully")