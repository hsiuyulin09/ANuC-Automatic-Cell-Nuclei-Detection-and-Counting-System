from pathlib import Path
import cv2
import h5py
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

DEFAULT_MIN_DISTANCE = 90
DEFAULT_CELL_PROB_THRESHOLD = 0.3
DEFAULT_MIN_CELL_SIZE = 6400
DEFAULT_BUFFER = 3

def setup_folders(config):
    paths = config["paths"]
    final_result = Path(paths["final_result"])

    final_result.mldir(parents=True, exist_ok=True)
    print("post processing setup successful")

def post_processing(h5_path, output_folder, min_distance, cell_prob_threshold, min_cell_size, buffer):
    h5_path = Path(h5_path)
    output_folder = Path(output_folder)

    if not h5_path.exists():
        print("prediction result (h5 file) not exist")
        return
    
    h5_filename = h5_path.stem
    output_path = output_folder / h5_filename

    output_path.mkdir(parents=True, exist_ok=True)

    report_line =[]

    with h5py.File(h5_path, "r") as f:
        for img_name in f.key(): # f.keys()釣出f裡所有group名組成list
            print(f"processing image. {img_name}")

            group = f["img_name"]
            raw_image = group["raw_name"][:]
            prob_map = group["probability_map"][:]
            h, w = prob_map.shape # raw_img.shape = (H, W, 3), prob_map.shape = (H, W)

def batch_processing(config):