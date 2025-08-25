from src.data_utils import create_cv_yaml_files
from ultralytics import YOLO
from typing import Dict, Any
from ray import tune
import numpy as np
import os

def train_yolo(config: Dict[str, Any]) -> None:
    """Defines the trainable function for a single Ray Tune trial.

    This function is the core of the HPO process, executed by Ray Tune for each
    hyperparameter configuration (a 'trial'). It performs a full k-fold
    cross-validation for the given config. For each fold, it trains a new YOLO
    model and records its validation mAP. Finally, it calculates the median of
    these mAPs and reports it back to Ray Tune, which uses this metric to guide
    the optimization process. All intermediate training artifacts for each fold
    are saved within the trial's specific directory in 'ray_results'.

    Args:
        config (Dict[str, Any]): A dictionary containing the hyperparameter
            configuration for this specific trial, supplied by Ray Tune.

    Returns:
        None: This function does not return a value but reports its results
              to Ray Tune via the `tune.report()` API.
    """
    fold_yaml_paths = create_cv_yaml_files(
        config["original_data_yaml"],
        config["k_folds"],
        config["n_repeats"],
        config["seed"]
    )
    
    fold_metrics = []
    trial_dir = tune.get_trial_dir()

    for i, fold_yaml in enumerate(fold_yaml_paths):
        model = YOLO(config["model_variant"])
        fold_project_dir = os.path.join(trial_dir, f"fold_{i}")
        
        results = model.train(
            data=fold_yaml,
            project=fold_project_dir,
            name="fold_run",
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
            verbose=False
        )
        fold_metrics.append(results.box.map)

    median_ap = np.median(fold_metrics)
    tune.report(median_ap=median_ap)