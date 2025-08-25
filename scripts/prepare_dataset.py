from typing import List, Dict, Any
import random, shutil, yaml, os

def _copy_files(file_list: List[str], src_img_path: str, src_lbl_path: str, dest_path: str) -> None:
    """Copies a list of images and their corresponding labels to a destination.

    This is an internal helper function. It creates 'images' and 'labels'
    subdirectories within the destination path and copies the specified files.

    Args:
        file_list (List[str]): A list of image filenames to be copied.
        src_img_path (str): The path to the source directory of the images.
        src_lbl_path (str): The path to the source directory of the labels.
        dest_path (str): The base path for the destination directory.

    Returns:
        None
    """
    dest_img_path = os.path.join(dest_path, "images")
    dest_lbl_path = os.path.join(dest_path, "labels")
    os.makedirs(dest_img_path, exist_ok=True)
    os.makedirs(dest_lbl_path, exist_ok=True)

    for img_file in file_list:
        base_name, _ = os.path.splitext(img_file)
        lbl_file = base_name + ".txt"
        
        shutil.copy(os.path.join(src_img_path, img_file), os.path.join(dest_img_path, img_file))
        shutil.copy(os.path.join(src_lbl_path, lbl_file), os.path.join(dest_lbl_path, lbl_file))


def prepare_dataset(config: Dict[str, Any]) -> None:
    """Splits the full dataset and generates all necessary .yaml config files.

    This function performs the initial setup by splitting the data from
    'data/full_dataset' into a development set (for HPO and final training) and a
    final test set (for evaluation). It reads a reference 'dataset.yaml' inside
    the full_folder to get class information ('nc' and 'names') and generates
    three new .yaml files required for the pipeline.

    Args:
        base_path (str, optional): The root directory for all data operations.
            Defaults to "data".
        full_folder (str, optional): The name of the folder containing the complete
            initial dataset. Defaults to "full_dataset".
        test_split (float, optional): The proportion of the dataset to be used for the
            final test set (e.g., 0.1 for 10%). Defaults to 0.1.
        seed (int, optional): The random seed for shuffling to ensure reproducible
            splits. Defaults to 42.

    Returns:
        None
    """
    
    paths = config['paths']
    dataset_params = config['dataset']
    random.seed(dataset_params['random_seed'])
    
    base_path = paths['data_root']
    full_folder = paths['full_dataset']
    test_split = dataset_params['test_split_ratio']
    
    reference_yaml_path = os.path.join(base_path, full_folder, "dataset.yaml")

    with open(reference_yaml_path, 'r') as f:
        reference_config = yaml.safe_load(f)
    class_config = {'nc': reference_config['nc'], 'names': reference_config['names']}
    
    full_images_path = os.path.join(base_path, full_folder, "images")
    full_labels_path = os.path.join(base_path, full_folder, "labels")
    all_images = [f for f in os.listdir(full_images_path) if f.endswith(('.jpg', '.png'))]
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * (1 - test_split))
    dev_files = all_images[:split_idx]
    test_files = all_images[split_idx:]

    dev_path = os.path.join(base_path, paths['dev_dataset'])
    test_path = os.path.join(base_path, paths['final_test_dataset'])

    _copy_files(dev_files, full_images_path, full_labels_path, dev_path)
    _copy_files(test_files, full_images_path, full_labels_path, test_path)

    with open(os.path.join(base_path, paths['dev_dataset_yaml']), "w") as f:
        yaml.dump({'train': os.path.abspath(os.path.join(dev_path, "images")), **class_config}, f)
    with open(os.path.join(base_path, paths['dev_full_yaml']), "w") as f:
        yaml.dump({'train': os.path.abspath(os.path.join(dev_path, "images")), **class_config}, f)
    with open(os.path.join(base_path, paths['final_test_yaml']), "w") as f:
        yaml.dump({'test': os.path.abspath(os.path.join(test_path, "images")), **class_config}, f)
        
    print("Datasets successfully prepared and YAML files generated.")