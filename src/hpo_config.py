from typing import Dict, Any
from ray import tune
import os

def get_search_space(config: Dict[str, Any]) -> Dict[str, Any]:
    """Defines the hyperparameter search space for the Ray Tune optimization.

    This function constructs and returns a dictionary that outlines the tunable
    hyperparameters and fixed parameters for the HPO process. It dynamically
    discovers the available YOLO model variants (.pt files) from the
    'base_models' directory to include them in the search space.

    Args:
        None

    Returns:
        Dict[str, Any]: A dictionary where keys are parameter names and values
                        are Ray Tune search space objects (e.g., tune.choice)
                        or fixed literal values.

    Raises:
        FileNotFoundError: If no .pt model files are found in the
                           'base_models' directory.
    """
    
    paths = config['paths']
    hpo_params = config['hpo']
    training_params = config['training']
    
    base_model_dir = paths['base_models']
    available_models = [os.path.join(base_model_dir, f) for f in os.listdir(base_model_dir) if f.endswith('.pt')]
    if not available_models:
        raise FileNotFoundError(f"No .pt model files found in '{base_model_dir}'.")

    search_space = {
        "model_variant": tune.choice(available_models),
        "seed": tune.randint(0, 1000),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "momentum": tune.uniform(0.85, 0.95),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "hsv_h": tune.uniform(0.0, 0.1), "hsv_s": tune.uniform(0.0, 0.9), "hsv_v": tune.uniform(0.0, 0.9),
        
        "original_data_yaml": os.path.join(paths['data_root'], paths['dev_dataset_yaml']),
        "epochs": training_params['epochs'],
        "patience": training_params['patience'],
        "batch_size": training_params['batch_size'],
        "imgsz": training_params['imgsz'],
        
        "k_folds": hpo_params['k_folds'],
        "n_repeats": hpo_params['n_repeats'],
    }
    return search_space