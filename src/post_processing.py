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
DEFAULT_SIGMA = 30
DEFAULT_MIN_CELL_SIZE = 6400
DEFAULT_BUFFER = 3

def setup_folders(config):
    paths = config["paths"]
    final_result = Path(paths["final_result"])

    final_result.mkdir(parents=True, exist_ok=True)
    print("post processing setup successful")

def post_processing(h5_path, output_folder, cell_prob_threshold, sigma, min_distance, min_cell_size, buffer):
    h5_path = Path(h5_path)
    output_folder = Path(output_folder)

    if not h5_path.exists():
        print("prediction result (h5 file) not exist")
        return
    
    h5_filename = h5_path.stem
    output_path = output_folder / h5_filename

    output_path.mkdir(parents=True, exist_ok=True)

    report_lines =[]

    with h5py.File(h5_path, "r") as f:
        for img_name in f.keys(): # f.keys()釣出f裡所有group名組成list
            print(f"processing image. {img_name}")

            group = f[img_name]
            raw_image = group["raw_image"][:]
            prob_map = group["probability_map"][:]
            h, w = prob_map.shape # raw_img.shape = (H, W, 3), prob_map.shape = (H, W)

            thresh = prob_map >= cell_prob_threshold

            distance = ndi.distance_transform_edt(thresh)
                # Euclidean Distance Transform, EDT
                # 算每一個True與最近的False的EDT距離, 存成matrix
            dist_smoothed = gaussian_filter(distance, sigma=sigma)
            coords = peak_local_max(dist_smoothed, min_distance=min_distance, labels=thresh)
                # peak_local_max(目標矩陣, 最短距離threshold, 區域限制)

            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True # 將細胞中心座標標成 boolean matrix
            markers, _ = ndi.label(mask) # 依順序編號, 標成 watershed seed
            labels = watershed(-distance, markers, mask=thresh)

            for label_id in np.unique(labels):
                if label_id == 0:
                    continue

                cell_area = np.sum(labels == label_id)

                if cell_area < min_cell_size:
                    labels[labels == label_id] = 0

            complete_count = 0
            incomplete_count = 0
            overlay_img = raw_image.copy()

            for label_id in np.unique(labels):
                if label_id == 0:
                    continue

                cell_mask = (labels == label_id)
                y_coords, x_coords = np.where(cell_mask)  # np.where() 用y, x回傳所有非0數值的座標, np 格式 y_coords = array([]), x_coords = array([])

                on_border = ( # 邊界計算
                    np.any(y_coords <= buffer)
                    or np.any(y_coords >= (h - buffer))
                    or np.any(x_coords <= buffer)
                    or np.any(x_coords >= (w - buffer))
                )

                if on_border:
                    incomplete_count += 1
                    labels[labels == label_id] = 0
                
                else:
                    complete_count += 1
                    color = (0, 255, 0)
                    contours, _ = cv2.findContours(cell_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # contours, hierarchy = cv2.findContours(目標矩陣, 模式, 儲存方式)
                        # contours 輪廓座標, hierarchy 輪廓內外關係
                    cv2.drawContours(overlay_img, contours, -1, color, 2)
                        # cv2.drawContours(畫布, 座標, 指定繪製目標contourIdx, 顏色, 線條寬度) 繪製輪廓
                        # contourIdx=-1 參數-1表示繪製全部

            thresh = labels > 0
            lower_bound = cell_prob_threshold
            upper_bound = np.max(prob_map)

            if upper_bound > lower_bound:
                norm_prob = np.clip((prob_map - lower_bound) / (upper_bound - lower_bound), 0, 1) # norm_prob 新標準化矩陣
            else:
                norm_prob = np.zeros_like(prob_map)

            heatmap = cv2.applyColorMap((norm_prob * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
                # cv2.applyColorMap(8-bit亮度圖矩陣, 選擇填色模式) 填色工具 #(norm_prob * 255) 映射成8 bit的0到255亮度

            mask_3ch = np.repeat(thresh[:, :, None], 3, axis=2)
                # np.repeat(目標矩陣, 重複次數, 目標維度) 複製維度 #目標維度數字*重複次數=結果, ex.np.repeat((1, 2), 3, axis=1) >>> shape = (1, 6)
            heatmap_overlay = np.where(mask_3ch, heatmap, raw_image)  # input shape = (H, W, 3) #np.where(boolean矩陣, true時填入, false時填入)

            img_stem = Path(img_name).stem
            cv2.imwrite(str(output_path / f"{img_stem}_counter.png"), overlay_img)
            cv2.imwrite(str(output_path / f"{img_stem}_heatmap.png"), heatmap_overlay)
            cv2.imwrite(str(output_path / f"{img_stem}_origin.png"), raw_image)

            total = complete_count + incomplete_count # 暫時保留
            report_lines.append(f"{img_name}: total {complete_count} cells")

    report_path = output_path / f"{h5_filename}_report.txt"

    with report_path.open("w", encoding="utf-8") as txt_file:
        txt_file.write("\n".join(report_lines))

    print(f"post-processing complete: {output_path}")

def batch_post_process(config):
    paths = config["paths"]
    postprocessing_config = config.get("postprocessing") or {}

    h5_path = Path(paths["prediction_results"])
    output_path = Path(paths["final_result"])

    cell_prob_threshold = postprocessing_config.get("cell_prob_threshold")
    if cell_prob_threshold is None:
        cell_prob_threshold = DEFAULT_CELL_PROB_THRESHOLD

    sigma = postprocessing_config.get("sigma")
    if sigma is None:
        sigma = DEFAULT_SIGMA

    min_distance = postprocessing_config.get("min_distance")
    if min_distance is None:
        min_distance = DEFAULT_MIN_DISTANCE

    min_cell_size = postprocessing_config.get("min_cell_size")
    if min_cell_size is None:
        min_cell_size = DEFAULT_MIN_CELL_SIZE

    buffer = postprocessing_config.get("buffer")
    if buffer is None:
        buffer = DEFAULT_BUFFER

    setup_folders(config)

    if not h5_path.exists():
        print("prediction result folder not exist")
        return
    
    h5_files = sorted(h5_path.glob("*.h5"))
    
    if not h5_files:
        print("prediction result not exist")
        return

    print(f"total files. {len(h5_files)}")

    for h5_file in h5_files:
        print(f"processing file. {h5_file.name}")
        post_processing(
            h5_file, output_path,
            cell_prob_threshold=cell_prob_threshold, sigma=sigma, min_distance=min_distance, min_cell_size=min_cell_size, buffer=buffer
            )

    print("post-processing complete.")        
