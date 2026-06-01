from fastapi import FastAPI, HTTPException
from api.schemas import HealthResponse, PredictRequest, PredictResponse
from api.service import DEFAULT_API_CONFIG, load_api_config, run_prediction_from_api_config

def create_app(api_config_path=DEFAULT_API_CONFIG): # 建立 FastAPI app
    api_config = load_api_config(api_config_path)
    api_info = api_config.get("api") or {}

    app = FastAPI(
        title=api_info.get("title", "ANuC Local API Server"),
        version=api_info.get("version", "2.0.0"),
    )

    @app.get("/health", response_model=HealthResponse) # 定義 API endpoint
    def health_check():
        status = {"status": "ok"}
        return status

    @app.post("/predict", response_model=PredictResponse)
    def run_prediction(request: PredictRequest | None = None):
        request = request or PredictRequest()

        try:
            return run_prediction_from_api_config(
                api_config_path=request.api_config_path or api_config_path,
                prediction_config_path=request.prediction_config_path,
                postprocessing_config_path=request.postprocessing_config_path,
            )

        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        except KeyError as error:
            raise HTTPException(status_code=400, detail=f"missing config key: {error}") from error

        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


app = create_app()
