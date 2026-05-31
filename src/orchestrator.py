def run_prediction_pipeline(prediction_config, postprocessing_config):

    from src.predict import prediction_model
    from src.post_processing import batch_post_process

    print("prediction pipeline start.")

    print("[1/2] running model prediction")
    prediction_model(prediction_config)

    print("[2/2] running post-processing")
    batch_post_process(postprocessing_config)

    print("prediction pipeline complete.")