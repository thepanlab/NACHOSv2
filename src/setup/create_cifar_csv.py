import os
import sys

import torchvision
from torch.utils.data import ConcatDataset
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from src.data_processing.split_dataset import split_dataset
from src.data_processing.create_cifar10_csv_and_image import create_cifar10_csv_and_image


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



def create_cifar_csv(dataset, number_of_splits, prefix, create_images = True):
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



if __name__ == "__main__":
    """
    Main entry point of the script. Loads the CIFAR-10 dataset and creates CSV files for dataset splits.
    """

    # Loads the CIFAR-10 dataset
    full_dataset = load_cifar10()
    
    # Creates 10 CSV files from the CIFAR-10 dataset splits
    create_cifar_csv(full_dataset, number_of_splits = 10, prefix = 'c', create_images = True)
