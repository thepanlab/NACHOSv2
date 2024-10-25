import json
import os
import pandas as pd
import sys

import torchvision
from torch.utils.data import ConcatDataset
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from src.data_processing.split_dataset import split_dataset
from src.image_processing.image_saver import save_image


def load_cifar10():
    """
    Loads the CIFAR-10 dataset and applies transformations.

    Returns:
        full_dataset (ConcatDataset): Combined training and test dataset.
    """
    
    # Defines transformation to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])
    
    
    # Loads training and test datasets with transformations
    trainset = torchvision.datasets.CIFAR10(root = f'{PROJECT_ROOT}/data', train = True, download = True, transform = transform)
    testset = torchvision.datasets.CIFAR10(root = f'{PROJECT_ROOT}/data', train = False, download = True, transform = transform)
    
    
    # Combines the training and test datasets
    full_dataset = ConcatDataset([trainset, testset])
    
    return full_dataset



def create_cifar10_csv_and_image(dataset, indices, filename, image_directory, create_images = True):
    """
    Creates a CSV file containing filenames and labels of images in a dataset split.

    Args:
        dataset (Dataset): The dataset from which to extract images.
        indices (list): The indices of the images to include in the CSV.
        filename (str): The filename for the CSV file.
        image_directory (str): The directory where the images will be saved.
        create_images (bool): Whether to save the images to the disk. Default is True.
    """
    
    # Initializes a list to hold the data rows
    data = []
    
    # Defines the columns for the CSV file
    columns = ['files', 'labels']
    

    for index in indices:
        
        # Extracts image and label for the current index
        image, label = dataset[index]
        
        # Generates a filename for the image
        image_filename = f"cifar_{index}.png"
        
        
        # Saves the image to the specified directory
        if create_images:
            save_image(image, os.path.join(image_directory, image_filename))
        
        
        # Creates a row with the image filename and label
        row = [image_filename, label]
        data.append(row)
    
    
    # Creates a DataFrame and saves it as a CSV file
    df = pd.DataFrame(data, columns = columns)
    
    file_directory = f'{PROJECT_ROOT}/data/cifar10_csv'
    
    os.makedirs(file_directory, exist_ok = True)
    
    
    df.to_csv(os.path.join(file_directory, filename), index = False)
    


def create_cifar10_dataset(dataset, number_of_splits, prefix, create_images = True):
    """
    Creates multiple CSV files from CIFAR-10 dataset splits.

    Args:
        dataset (ConcatDataset): The dataset to split and save as CSV.
        number_of_splits (int): The number of splits to create.
        prefix (str): The prefix for the filenames of the CSV files.
        create_images (bool): Whether to create image files for each split. Default is True.
    """
    
    # Splits the dataset into the specified number of splits
    splits = split_dataset(dataset, number_of_splits)
    
    
    # Creates a directory for the images if it doesn't exist
    image_directory = os.path.join('data', 'cifar10_images')
    os.makedirs(image_directory, exist_ok = True)
    
    
    # Iterates over the splits and create a CSV for each
    for i, split_indices in enumerate(splits, 1):
        
        create_cifar10_csv_and_image(dataset, split_indices, f"{prefix}{i}.csv", image_directory, create_images)

    
    
def create_cifar10_config_file():
    config = {
        "hyperparameters": {
            "batch_size": 32,
            "channels": 1,
            "cropping_position": [40, 10],
            "decay": 0.01,
            "do_cropping": False,
            "epochs": 20,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "bool_nesterov": True,
            "patience": 10
        },
        
        "data_input_directory": "/data/cifar10_images",
        "csv_input_directory": "data/cifar10_csv",
        "output_path": "results/distributed/cifar_test",
        "job_name": "cifar_test",
        
        "k_epoch_checkpoint_frequency": 1,
        
        "shuffle_the_images": True,
        "shuffle_the_folds": False,
        "seed": 9,
        
        "class_names": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        "selected_model_name": "Cifar10FF",
        "metrics": [],
        
        "subject_list": ["cifar_1", "cifar_2", "cifar_3", "cifar_4", "cifar_5", "cifar_6", "cifar_7", "cifar_8", "cifar_9", "cifar_10"],
        "test_subjects": ["cifar_1", "cifar_2", "cifar_3", "cifar_4", "cifar_5", "cifar_6", "cifar_7", "cifar_8", "cifar_9", "cifar_10"],
        "validation_subjects": ["cifar_1", "cifar_2", "cifar_3", "cifar_4", "cifar_5", "cifar_6", "cifar_7", "cifar_8", "cifar_9", "cifar_10"],
        
        "image_size": [32, 32],
        "target_height": 301,
        "target_width": 235
    }
    
    with open(os.path.join(f'{PROJECT_ROOT}/scripts/config_files', "cifar_test_config_inner_inceptionV3_trial_parallel.json"), 'w') as f:
        json.dump(config, f, indent=4)



if __name__ == "__main__":
    """
    Main entry point of the script. Loads the CIFAR-10 dataset and creates CSV files for dataset splits.
    """

    # Loads the CIFAR-10 dataset
    full_dataset = load_cifar10()
    
    # Creates 10 CSV files from the CIFAR-10 dataset splits
    create_cifar10_dataset(full_dataset, number_of_splits = 10, prefix = 'cifar_', create_images = True)

    # Creates the configuration file
    create_cifar10_config_file()
    