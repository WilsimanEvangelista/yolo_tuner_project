from ultralytics import YOLO
from typing import Dict, Any
import json
import os

def train_final_model(config_path: str, config: Dict[str, Any]) -> str:
    """Trains the final production model using the best hyperparameters.

    This function loads the optimal hyperparameter configuration found during the
    HPO phase from a JSON file. It then initializes a new YOLO model and trains
    it from scratch on the entire development dataset ('dev_full.yaml'). This
    produces the final model artifact that will be used for the final evaluation
    and potential deployment.

    Args:
        config_path (str): The file path to the 'best_config.json' which
            contains the optimal hyperparameters.

    Returns:
        str: The path to the directory where the final trained model and its
             artifacts (weights, graphs, etc.) are saved.
    """
    
    paths = config['paths']
    
    with open(config_path, "r") as f:
        config = json.load(f)

    model = YOLO(config["model_variant"])

    results = model.train(
        data=os.path.join(paths['data_root'], paths['dev_full_yaml']),
        epochs=config["epochs"],
        patience=config["patience"],
        batch=config["batch_size"],
        imgsz=config["imgsz"],
        seed=config["seed"],
        lr0=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"],
        hsv_v=config["hsv_v"],
        verbose=True,
        project=paths["production_model"],
        name="final_yolo_model"
    )
    
    return results.save_dir