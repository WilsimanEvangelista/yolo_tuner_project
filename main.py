from src.evaluate_on_final_test_set import evaluate_on_final_test_set
from scripts.prepare_dataset import prepare_dataset
from src.train_final_model import train_final_model
from src.run_tuning import run_tuning_phase
from src.utils import load_config
import os

def main() -> None:
    """Orchestrates the entire machine learning pipeline.

    This main script executes the full workflow in a sequential and robust manner:
    1.  Prepares the datasets by splitting the data and creating YAML files.
    2.  Runs the hyperparameter optimization (HPO) phase to find the best model
        architecture and training parameters.
    3.  Trains the final production model using the best configuration found.
    4.  Performs a final, impartial evaluation on a blind test set.

    Args:
        None

    Returns:
        None
    """
    
    config = load_config()
    separator = "=" * 20

    print(f"{separator}\nSTEP 0: PREPARING DATASETS\n{separator}")
    prepare_dataset(config)
    
    print(f"\n{separator}\nSTEP 1: HYPERPARAMETER OPTIMIZATION (HPO)\n{separator}")
    best_config_path = run_tuning_phase(config)

    if os.path.exists(best_config_path):
        print(f"\n{separator}\nSTEP 1 COMPLETE. Best config saved to {best_config_path}\n{separator}")
        
        print(f"\n{separator}\nSTEP 2: TRAINING FINAL MODEL\n{separator}")
        final_model_dir = train_final_model(best_config_path, config)
        final_weights_path = os.path.join(final_model_dir, "weights/best.pt")

        if os.path.exists(final_weights_path):
            print(f"\n{separator}\nSTEP 2 COMPLETE. Final model saved in {final_model_dir}\n{separator}")
            
            print(f"\n{separator}\nSTEP 3: FINAL AND IMPARTIAL EVALUATION\n{separator}")
            evaluate_on_final_test_set(final_weights_path, config)
            
            print(f"\n{separator}\nENTIRE PIPELINE COMPLETED SUCCESSFULLY!\n{separator}")
        else:
            print(f"\n[ERROR] Final training failed. Weights not found at '{final_weights_path}'.")
    else:
        print(f"\n[ERROR] HPO phase failed. '{config['paths']['best_config_json']}' was not created.")

if __name__ == "__main__":
    main()