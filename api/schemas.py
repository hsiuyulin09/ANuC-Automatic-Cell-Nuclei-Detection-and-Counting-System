from typing import Optional
from pydantic import BaseModel # 資料驗證

class PredictRequest(BaseModel): 
        # BaseModel 建立資料格式
        # 檢查 user 的 config path 輸入格式
    api_config_path: Optional[str] = None
    prediction_config_path: Optional[str] = None
    postprocessing_config_path: Optional[str] = None


class PredictResponse(BaseModel): # 定義回傳格式
    status: str
    message: str
    prediction_results: str
    final_result: str


class HealthResponse(BaseModel):
    status: str
