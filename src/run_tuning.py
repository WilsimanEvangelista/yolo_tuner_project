from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from src.hpo_config import get_search_space
from src.trainable_yolo import train_yolo
from typing import Dict, Any
from ray import tune
import json

def run_tuning_phase(config: Dict[str, Any]) -> str:
    """Executes the hyperparameter optimization (HPO) phase using Ray Tune.

    This function sets up and runs the main Ray Tune experiment. It uses the
    search space from `hpo_config`, the trainable function `train_yolo`, the
    Optuna search algorithm for Bayesian optimization, and the ASHAScheduler
    for early stopping of unpromising trials. After the run completes, it
    identifies the best hyperparameter configuration based on the 'median_ap'
    metric and saves it to a 'best_config.json' file.

    Args:
        None

    Returns:
        str: The file path to the saved 'best_config.json' file.
    """
    metric = config["hpo"]["metric"]
    mode = config["hpo"]["mode"]
    
    search_space = get_search_space()
    
    search_algorithm = OptunaSearch(metric=metric, mode=mode)
    
    scheduler = ASHAScheduler(metric=metric, mode=mode, grace_period=1)
    
    analysis = tune.run(
        train_yolo,
        config=search_space,
        search_alg=search_algorithm,
        scheduler=scheduler,
        num_samples=20,
        resources_per_trial=config["hpo"]["resources_per_trial"],
        name=config["hpo"]["experiment_name"],
        local_dir=config["paths"]["ray_results"]
    )
    
    best_config = analysis.get_best_config(metric=metric, mode=mode)
    
    output_path = config["paths"]["best_config_json"]
    with open(output_path, "w") as f:
        json.dump(best_config, f, indent=2)
        
    return output_path