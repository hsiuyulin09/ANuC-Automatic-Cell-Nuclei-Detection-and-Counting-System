import random
from pathlib import Path
import cv2
import h5py
import numpy as np

DEFAULT_SAMPLE_RATE_EMPTY = 0.2
DEFAULT_DATASET_SEED = 77
DEFAULT_SPLIT = [0.9, 0.1]

def create_h5_dataset(config):
    paths = config["paths"]
    dataset_config = config.get("dataset") or {}

    # required paths
    image_folder = Path(paths["labelme_input"])
    mask_folder = Path(paths["masks_output"])
    output_h5 = Path(paths["dataset_output"])

    sample_rate_empty = dataset_config.get("sample_rate_empty")
    if sample_rate_empty is None:
        sample_rate_empty = DEFAULT_SAMPLE_RATE_EMPTY

    seed = dataset_config.get("seed")
    if seed is None:
        seed = DEFAULT_DATASET_SEED

    split = dataset_config.get("split")
    if split is None:
        split = DEFAULT_SPLIT

    for folder in [image_folder, mask_folder]:
        if not folder.exists():
            print(f"{folder} not exist")
            return
        
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    # 樣本分類 (positive, empty)
    image_paths = sorted(image_folder.glob("*.png"))
    positive_samples = []
    empty_samples = []

    for image_path in image_paths: # 根據圖片檔名, 找出它對應的 mask 檔案路徑
        mask_path = mask_folder / f"{image_path.stem}_mask.png"
        
        if mask_path.exists():
            mask_data = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask_data is None:
                raise RuntimeError(f"failed to read mask: {mask_path}")

            if np.sum(mask_data) == 0:
                empty_samples.append(image_path)
            else:
                positive_samples.append(image_path)

    print(f"total positive {len(positive_samples)}, total empty {len(empty_samples)} \n total {len(positive_samples)+len(empty_samples)}")

    num_empty_keep = int(len(positive_samples)*sample_rate_empty)
    num_empty_keep = min(num_empty_keep, len(empty_samples))
    selected_empty_samples = random.sample(empty_samples, num_empty_keep) if empty_samples else [] # random.sample(清單, 數量)不重複隨機抽取

    total_samples = positive_samples + selected_empty_samples
    random.shuffle(total_samples)

    if not total_samples:
        print(".png files not exist")
        return
    
    print(f"dataset composition. cell positive image {len(positive_samples)}, cell empty image {len(empty_samples)} \n total image {len(total_samples)}")

    total = len(total_samples)
    train_end = int(total * split[0])

    datasets = {
        "train": total_samples[:train_end], 
        "val": total_samples[train_end:]
        }

    with h5py.File(output_h5, "w") as hf: # h5py.File(output_h5, "w") 建立或覆寫 H5 檔案
        for group_name, file_list in datasets.items():
            # datasets.items() 會取出 group_name 和 file_list
            # 例：group_name="train", file_list=[Path(...), ...]
            count = len(file_list)

            if count == 0:
                continue

            sample_img = cv2.imread(str(file_list[0]))
            if sample_img is None:
                print(f"failed to read image: {file_list[0]}")
                continue

            h, w, c = sample_img.shape

            # 建立 H5 group 和 datasets： /train/images, /train/masks, /val/images, /val/masks
            group = hf.create_group(group_name)
            image_ds = group.create_dataset("images", shape=(count, h, w, c), dtype="uint8")
            mask_ds = group.create_dataset("masks", shape=(count, h, w), dtype="uint8")

            for i, image_path in enumerate(file_list): # 實際讀取和寫入數據, 前面是建立 dataset
                mask_path = mask_folder/f"{image_path.stem}_mask.png"

                img_data = cv2.imread(str(image_path))
                mask_data = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                if img_data is None:
                    raise RuntimeError(f"failed to read image: {image_path}")

                if mask_data is None:
                    raise RuntimeError(f"failed to read mask: {mask_path}")
                
                image_ds[i] = img_data
                mask_ds[i] = mask_data

                print(image_path.name)

            print(f"{group_name}: {count} data")

    print(f"{output_h5} create complete")
