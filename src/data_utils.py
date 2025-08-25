from sklearn.model_selection import RepeatedKFold
from typing import List
import numpy as np
import yaml
import os

def create_cv_yaml_files(original_yaml_path: str, k_folds: int, n_repeats: int, seed: int) -> List[str]:
    """Generates temporary .yaml files for each fold of a RepeatedKFold CV.

    This function reads a main dataset YAML file, finds all associated images,
    and then uses scikit-learn's RepeatedKFold to split them into training and
    validation sets for each fold. It writes a new, temporary .yaml file for
    each of these folds to a 'temp_yamls' directory.

    Args:
        original_yaml_path (str): Path to the main dataset .yaml file. This
            file provides the path to the images and class information.
        k_folds (int): The number of folds (k) for cross-validation.
        n_repeats (int): The number of times the cross-validation is repeated.
        seed (int): The random seed for reproducible splits.

    Returns:
        List[str]: A list of absolute paths to the generated .yaml files,
            one for each fold.
    """
    with open(original_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    data_root = os.path.dirname(original_yaml_path)
    images_path = os.path.abspath(os.path.join(data_root, data_config['train']))
    
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
    all_images_paths = np.array([os.path.join(images_path, f) for f in image_files])

    rkf = RepeatedKFold(n_splits=k_folds, n_repeats=n_repeats, random_state=seed)
    
    temp_yaml_dir = os.path.abspath('./temp_yamls')
    os.makedirs(temp_yaml_dir, exist_ok=True)
    
    generated_yaml_paths = []
    
    for i, (train_indices, val_indices) in enumerate(rkf.split(all_images_paths)):
        train_files = all_images_paths[train_indices].tolist()
        val_files = all_images_paths[val_indices].tolist()

        fold_config = {
            'train': train_files,
            'val': val_files,
            'names': data_config['names'],
            'nc': data_config['nc']
        }
        
        fold_yaml_path = os.path.join(temp_yaml_dir, f'fold_{i}.yaml')
        
        with open(fold_yaml_path, 'w') as f:
            yaml.dump(fold_config, f)
            
        generated_yaml_paths.append(fold_yaml_path)
        
    return generated_yaml_paths