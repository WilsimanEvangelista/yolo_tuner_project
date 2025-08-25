from ultralytics import YOLO
from typing import Dict, Any
import os

def evaluate_on_final_test_set(weights_path: str, config: Dict[str, Any]) -> str:
    """Evaluates the final trained model on the blind final test set.

    This function loads the production model's weights and runs a final,
    impartial evaluation on the test data that was held out during the
    entire HPO and training process. It prints the key performance metrics
    and returns the directory where results were saved.

    Args:
        weights_path (str): The path to the final model's weights file
            (e.g., 'best.pt').
        final_test_yaml (str): The path to the .yaml file that defines the
            final test dataset.

    Returns:
        str: The path to the directory where Ultralytics saved the evaluation
             results (graphs, confusion matrix, etc.).
    """
    paths = config['paths']
    final_test_yaml_path = os.path.join(paths['data_root'], paths['final_test_yaml'])
    
    print(f"\n--- Starting Final Evaluation on the Blind Test Set ---")
    model = YOLO(weights_path)
    
    metrics = model.val(data=final_test_yaml_path, split='test')
    
    print("\n--- Final and Impartial Results ---")
    
    print("\nOverall Performance (Primary Metric):")
    print(f"  - mAP50-95: {metrics.box.map:.4f}")
    print(f"  - mAP50:    {metrics.box.map50:.4f}")
    print(f"  - mAP75:    {metrics.box.map75:.4f}")

    print("\nMetrics at Optimal F1-Score Threshold:")
    print(f"  - F1-Score:  {metrics.box.f1[0]:.4f}")
    print(f"  - Precision: {metrics.box.p[0]:.4f}")
    print(f"  - Recall:    {metrics.box.r[0]:.4f}")
    
    print(f"\nFinal evaluation reports and graphs saved in: {metrics.save_dir}")

    return metrics.save_dir