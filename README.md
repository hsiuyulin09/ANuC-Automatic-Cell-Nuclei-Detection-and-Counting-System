# 自動化免疫螢光染色細胞核偵測及計數系統
 Automatic Cell Nuclei Detection and Counting System

## 簡介

本專案提供自影像前處理、人工標註轉換、模型訓練、模型預測至後處理計數之完整流程。主要用於分析免疫螢光分析 (Immunofluorescence assay, IFA) 中的 DAPI channel 螢光顯微影像。透過 CLAHE 影像增強、Sliding Window 影像切割、UNet model 與 Watershed algorithm，完成細胞核區域偵測、分割、視覺化與計數輸出。

 - 版本簡述

    version 2.0.0

    ```text
    - CLI 操作系統
    - Local API Server
    - System decoupling
    ```

## 結果範例

<table>
  <tr>
    <td><img src="./pictures/20211224_origin.png" width="190" alt="Original image"></td>
    <td><img src="./pictures/20211224_heatmap.png" width="190" alt="Heatmap"></td>
    <td><img src="./pictures/20211224_counter.png" width="190" alt="Contour overlay"></td>
    <td><img src="./pictures/point1.png" width="190" alt="Zoom-in view"></td>
  </tr>
  <tr>
    <td align="center">FIG. 1-1 Original Image</td>
    <td align="center">FIG. 1-2 Heatmap</td>
    <td align="center">FIG. 1-3 Contour Overlay</td>
    <td align="center">FIG. 1-4 Zoom-in View</td>
  </tr>
</table>

## 目錄

- [簡介](#簡介)
- [結果範例](#結果範例)
- [專案特色](#專案特色)
- [檔案結構](#檔案結構)
- [環境需求](#環境需求)
- [快速開始](#快速開始)
  - [1. 系統初始化](#1-系統初始化)
  - [2. 放入待分析影像](#2-放入待分析影像)
  - [3. 開始分析](#3-開始分析)
  - [4. 查看輸出結果](#4-查看輸出結果)
- [詳細使用說明](#詳細使用說明)
  - [Train mode](#train-mode)
  - [Predict mode](#predict-mode)
- [Local API Server](#local-api-server)
  - [API Config](#api-config)
  - [啟動 API Server](#啟動-api-server)
  - [Health Check](#health-check)
  - [執行 Prediction mode](#執行-prediction-mode)
- [技術整合說明](#技術整合說明)
  - [Image Preprocessing](#image-preprocessing)
  - [Mask Generation](#mask-generation)
  - [數據封裝](#數據封裝)
  - [Model](#model)
  - [Loss Function](#loss-function)
  - [Device Selection](#device-selection)
  - [Post-processing](#post-processing)
- [特別感謝](#特別感謝)
- [關於作者](#關於作者)

## 專案特色

 - CLI 與 Local API Server 作為主要使用介面
 - Local API Server 提供 `GET /health` 與 `POST /predict`，讓地端 agent、GUI 或其他程式可以透過 HTTP 呼叫
 - Post processing 輸出 origin image、heatmap、counter overlay 與文字報告

## 檔案結構

```text
.
├─ main.py
├─ requirements.txt
├─ README.md
├─ configs/
│  ├─ preprocessing_config.yaml
│  ├─ training_config.yaml
│  ├─ prediction_config.yaml
│  ├─ postprocessing_config.yaml
│  └─ api_config.yaml
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ image_preprocessing.py
│  ├─ labelme_masks.py
│  ├─ dataset_h5.py
│  ├─ model.py
│  ├─ train.py
│  ├─ predict.py
│  ├─ post_processing.py
│  └─ orchestrator.py
├─ api/
│  ├─ __init__.py
│  ├─ schemas.py
│  ├─ service.py
│  └─ server.py
├─ pictures/
│  └─ ...
└─ unet_cellcount_model.pth
```

## 環境需求

 - 開發環境使用 `Python 3.12`

 - 建立虛擬環境

    - 建立 `Python 3.12` 環境

        ```bash
        conda create -n anuc python=3.12 -y
        ```

    - 啟動環境
    
        ```bash
        conda activate anuc
        ```

 - 安裝套件

    - 一鍵安裝

        ```bash
        pip install "opencv-python>=4.9" "numpy>=1.26" "h5py>=3.10" "PyYAML>=6.0" torch torch-directml matplotlib scipy scikit-image fastapi pydantic uvicorn
        ```

    - 或使用預寫好的 `requirements.txt` 安裝

        ```bash
        pip install -r requirements.txt
        ```

 - 安裝 Labelme

    如要自行建立 train set 需下載開源標記軟體 Labelme 並手動標記

    ```bash
    pip install labelme
    ```
## 快速開始

使用並載入專案內提供的預訓練權重紀錄，直接使用 prediction 模式進行影像辨識功能。

以 PowerShell CLI 介面執行 :

### 1. 系統初始化

```bash
python main.py anuc init
```

### 2. 放入待分析影像

在根目錄下找到 `prediction_input/` 並將須分析的影像檔放入此目錄

### 3. 開始分析

```bash
python main.py anuc predict
```

### 4. 查看輸出結果

 - Model 輸出的機率圖以 .h5 格式輸出至 `prediction_results/`

 - 完成所有處理的結果會輸出至 `final_result/`

    - 輸出結果預期:

        ```text
        .
        └─ final_result/
            └─ {h5_filename}/
                ├─ {image_stem}_origin.png
                ├─ {image_stem}_heatmap.png
                ├─ {image_stem}_counter.png
                └─ {timestamp}_report.txt
        ```

## 詳細使用說明

 - `main.py` 入口支援 CLI 指令如下:

    ```bash
    python main.py init
    python main.py preprocessing
    python main.py mask
    python main.py package
    python main.py train run
    python main.py anuc init
    python main.py anuc predict
    ```

### Train mode

#### 1. 系統初始化

```bash
python main.py init
```

建立 `preprocessing_input/` 目錄

#### 2. 載入影像並完成前處理

將預計作為 training set 的原始影像放入 `preprocessing_input/` 後，執行：

```bash
python main.py preprocessing
```

 - 影像前處理
    
     - 技術 :

        - Bilateral Filter
        - CLAHE
        - Sliding Window


     - 輸出 :

        ```text
        preprocessing_output/{image_stem}_y{y}_x{x}.png
        preprocessing_check/enhanced_{original_image_name}
        ```

#### 3. Labelme 人工標記

 - 在終端開啟 Labelme

    ```bash
    labelme
    ```

 - 設定 Labelme 中 label 為 "cell"

 - Labelme 會於檔案來源目錄輸出 `JSON`

#### 4. Binary Mask 轉換

```bash
python main.py mask
```

 - 讀取 Labelme `.json`，將 polygon points 轉換為 binary mask

     - 技術：

        - Labelme JSON parsing
        - OpenCV
        - Binary mask generation


     - 輸出：

        ```text
        masks_output/{tile_stem}_mask.png
        masks_check/check_{tile_stem}.png
        ```

#### 5. 數據封裝

```bash
python main.py package
```

 - 整合 tile image 與對應 mask，並寫入 `.h5` dataset

     - 技術：

        - h5py
        - positive / empty sample selection
        - train / validation split

     - 輸出：

        ```text
        datasets/cell_dataset.h5
        ```

     - cell_dataset.h5 structure

        ```text
        {
            "train": {
                "images": "shape = (train_count, H, W, C), dtype = uint8",
                "masks": "shape = (train_count, H, W), dtype = uint8",
            },
            "val": {
                "images": "shape = (val_count, H, W, C), dtype = uint8",
                "masks": "shape = (val_count, H, W), dtype = uint8",
            },
        }
        ```

#### 6. 開始訓練

```bash
python main.py train run
```

 - 技術：

    - PyTorch
    - UNet
    - BCEWithLogitsLoss
    - DiceLoss
    - Adam optimizer

 - 輸出：

    ```text
    unet_cellcount_model.pth
    ```

### Predict mode

#### 1. 系統初始化

```bash
python main.py anuc init
```

建立 `prediction_input/` 目錄


#### 2. 載入影像

根目錄下找到 `prediction_input/` 並將須分析的影像檔放入此目錄

預設支援副檔名 `.jpg`, `.jpeg`, `.tif`, `.png`

#### 3. 開始分析

```bash
python main.py anuc predict
```

 - 技術：

    - Sliding Window Prediction
    - UNet
    - Weight mask overlap blending
    - Watershed Segmentation Algorithm

 - 輸出：

    ```text
    prediction_results/{timestamp}.h5
    final_result/{timestamp}/...
    ```

## Local API Server

API 目前僅支援 prediction mode

(目前 API 是 folder-based prediction API，不是 upload API。也就是使用者要先把影像放到 `prediction_input/`，再呼叫 `/predict`)

### API Config

預設 API 環境

`configs/api_config.yaml`：

```yaml
server:
  host: 127.0.0.1
  port: 8000

configs:
  prediction_config: configs/prediction_config.yaml
  postprocessing_config: configs/postprocessing_config.yaml

api:
  title: ANuC Local API Server
  version: 2.0.0
```
    
### 啟動 API Server

```bash
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

FastAPI 文件頁面：

```text
http://127.0.0.1:8000/docs
```

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

Request line / headers：

```http
GET /health HTTP/1.1
Host: 127.0.0.1:8000
Accept: application/json
```

Request body：

```text
(empty)
```

回傳內容：

```json
{
  "status": "ok"
}
```

### 執行 Prediction mode

以下 `curl` 指令主要用於開發或測試階段，模擬外部程式呼叫 API 的行為。

正式整合時，可由其他程式透過 HTTP request 呼叫相同 endpoint。

 - 不指定 config path：

    ```bash
    curl -X POST http://127.0.0.1:8000/predict
    ```

    Request line / headers：

    ```http
    POST /predict HTTP/1.1
    Host: 127.0.0.1:8000
    Accept: application/json
    Content-Type: application/json
    ```

    Request body：

    可空白

    ```text
    (empty)
    ```

 - 指定 config path

    Request body：

    ```json
    {
        "api_config_path": "configs/api_config.yaml",
        "prediction_config_path": "configs/prediction_config.yaml",
        "postprocessing_config_path": "configs/postprocessing_config.yaml"
    }
    ```

    欄位皆為 optional。

    user 未提供時，會使用 `api_config.yaml` 中設定的預設 config path。

    帶 config path 的 `curl` 測試指令：

    ```bash
    curl -X POST http://127.0.0.1:8000/predict ^
        -H "Content-Type: application/json" ^
        -H "Accept: application/json" ^
        -d "{\"prediction_config_path\":\"configs/prediction_config.yaml\",\"postprocessing_config_path\":\"configs/postprocessing_config.yaml\"}"
    ```

    回傳內容：

    ```json
    {
        "status": "complete",
        "message": "prediction pipeline complete",
        "prediction_results": "prediction_results",
        "final_result": "final_result"
    }
    ```

## 技術整合說明

### Image Preprocessing

 - Bilateral Filter
 - CLAHE
 - Sliding Window

### Mask Generation

- 取得 polygon points
- `cv2.fillPoly()` 建立 binary mask

#### 數據封裝

- 使用 H5 儲存 train / validation images 與 masks
- 讀取 H5 轉為 PyTorch tensor
- training mode 下支援旋轉與亮度 augmentation

#### Model

UNet：

- encoder
- max pooling
- decoder
- skip connection
- final convolution

#### Loss Function

```text
BCEWithLogitsLoss + DiceLoss
```

#### Device Selection

預設選擇順序：

1. NVIDIA CUDA
2. DirectML (AMD GPU)
3. CPU

#### Post-processing

- probability threshold
- Euclidean distance transform
- Gaussian smoothing
- peak local max
- watershed segmentation
- contour extraction
- output: heatmap / counter / report output



## 特別感謝
感謝 <a href="https://www.erixnet.com/">EriXNet</a> 對開發的協助與顧問工作

## 關於作者

林修渝 Hsiu-Yu, Lin

臺灣人，來自台南市。喜歡戰錘40k、喜歡音樂、喜歡騎車、喜歡一切亂七八糟對工作沒什麼幫助的事情。

生物化學碩士，曾任職中央研究院生醫所研究助理，現職 AI 生物資訊工程師。專長是生物化學、癌症細胞生物學、外泌體、生物醫學數據分析、LLM 應用部屬、RAG、Agent Skill。

Hsiu-Yu, Lin

A AI bioinformatic engineer hailing from Tainan, Taiwan. I’m passionate about Warhammer 40k, music, motorcycle touring, and baseball. Basically, anything and everything that has absolutely nothing to do with my job.

I am a Master of Biochemistry and a former Research Assistant at the Institute of Biomedical Science, Academia Sinica. Now, I work as an AI Biomedical Data Scientist, and specialize in biochemistry, cancer biology, and biomedical image analysis.

GitHub : https://github.com/hsiuyulin09
<br><br>

<span style="font-size: 12px; font-weight: bold;">
<img src="./pictures/Python.png"style="width: 7.6%;" alt="python logo">
<img src="./pictures/PyTorch_logo.png"style="width: 3.5%;" alt="pytorch logo">
&emsp;<img src="./pictures/Jupyter_logo.png"style="width: 3.9%;" alt="jupyter logo">
&emsp;<img src="./pictures/icon-256.png"style="width: 4%;" alt="labelme logo">
&emsp;<img src="./pictures/erix.jpg"style="width: 13%;" alt="erixnet logo">
