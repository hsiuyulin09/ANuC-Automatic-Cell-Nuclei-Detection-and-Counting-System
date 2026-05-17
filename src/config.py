from pathlib import Path
import yaml

def load_config(config_path="config.yaml"):
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError("config file cannot found")
    
    with config_file.open("r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    return config_yaml