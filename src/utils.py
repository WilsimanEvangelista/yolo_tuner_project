from typing import Dict, Any
import yaml

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads the main YAML configuration file.

    Args:
        config_path (str, optional): The path to the configuration file.
            Defaults to "config.yaml".

    Returns:
        Dict[str, Any]: A dictionary containing the project configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config