from pathlib import Path
import yaml

def load_config(config_path):
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"config file cannot found: {config_file}")
    
    with config_file.open("r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    return config_yaml
