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

    - 或是使用預寫好的 `requirements.txt` 安裝

        ```bash
        pip install -r requirements.txt
        ```

 - 安裝 Lebalme

    如要自行建立 train set 需下載開源標記軟體 Lebalme 並手動標記

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
    
     - 技術 (影像增強及切割) :

        ```text
        - Bilateral Filter
        - CLAHE
        - Sliding Window
        ```

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
    



## 特別感謝</span><br>
感謝 <a href="https://www.erixnet.com/">EriXNet</a> 對開發的協助與顧問工作<br><br>

## 關於作者<br>
林修渝 Hsiu-Yu, Lin</span><br>
<span style="font-size: 12px; font-weight: bold;">
臺灣人，來自台南市。喜歡戰錘40k、喜歡音樂、喜歡騎車、喜歡一切亂七八糟對工作沒什麼幫助的事情。<br>
生物化學碩士，曾任職中央研究院生醫所研究助理，現職 ai 生物資訊工程師。專長是生物化學、癌症細胞生物學、外泌體及生物醫學數據分析、。<br><br>
Hsiu-Yu, Lin</span><br>
A ai bioinformatic engineer hailing from Tainan, Taiwan. I’m passionate about Warhammer 40k, music, motorcycle touring, and baseball. Basically, anything and everything that has absolutely nothing to do with my job.<br>
I am a Master of Biochemistry and a former Research Assistant at the Institute of Biomedical Science, Academia Sinica. Now, I work as an AI Biomedical Data Scientist, and specialize in biochemistry, cancer biology, and biomedical image analysis.<br><br>
GitHub : https://github.com/hsiuyulin09
</span>
<br><br>

<span style="font-size: 12px; font-weight: bold;">
<img src="./pictures/Python.png"style="width: 7.6%;" alt="pyhton logo">
<img src="./pictures/PyTorch_logo.png"style="width: 3.5%;" alt="pytorch logo">
&emsp;<img src="./pictures/Jupyter_logo.png"style="width: 3.9%;" alt="jupter logo">
&emsp;<img src="./pictures/icon-256.png"style="width: 4%;" alt="labelme logo">
&emsp;<img src="./pictures/erix.jpg"style="width: 13%;" alt="erixnet logo">
