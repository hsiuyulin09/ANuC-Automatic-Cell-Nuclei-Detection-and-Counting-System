from src.config import load_config
from src.orchestrator import run_prediction_pipeline

DEFAULT_API_CONFIG = "configs/api_config.yaml"

def load_api_config(api_config_path=DEFAULT_API_CONFIG):
    config = load_config(api_config_path)
    return config


def resolve_pipeline_config_paths(
    api_config_path=DEFAULT_API_CONFIG,
    prediction_config_path=None,
    postprocessing_config_path=None,
):
    api_config = load_api_config(api_config_path)
    config_paths = api_config["configs"]

    resolved_prediction_config_path = prediction_config_path or config_paths["prediction_config"]
    resolved_postprocessing_config_path = postprocessing_config_path or config_paths["postprocessing_config"]

    return resolved_prediction_config_path, resolved_postprocessing_config_path


def run_prediction_from_api_config(
    api_config_path=DEFAULT_API_CONFIG,
    prediction_config_path=None,
    postprocessing_config_path=None,
):
    resolved_prediction_config_path, resolved_postprocessing_config_path = resolve_pipeline_config_paths(
        api_config_path=api_config_path,
        prediction_config_path=prediction_config_path,
        postprocessing_config_path=postprocessing_config_path,
    )

    prediction_config = load_config(resolved_prediction_config_path)
    postprocessing_config = load_config(resolved_postprocessing_config_path)

    run_prediction_pipeline(
        prediction_config=prediction_config,
        postprocessing_config=postprocessing_config,
    )

    response = {
        "status": "complete",
        "message": "prediction pipeline complete",
        "prediction_results": prediction_config["paths"]["prediction_results"],
        "final_result": postprocessing_config["paths"]["final_result"],
    }

    return response
