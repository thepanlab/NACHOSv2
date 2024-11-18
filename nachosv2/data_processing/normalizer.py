from typing import Callable
from termcolor import colored

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from nachosv2.data_processing.calculate_mean_std import calculate_mean_std
from nachosv2.file_processing.filter_files import filter_files_by_extension
from nachosv2.data_processing.full_dataset import FullDataset
from nachosv2.file_processing.get_from_directory import get_files_from_directory
from nachosv2.data_processing.normalize_transform import NormalizeTransform


def get_mean_std(dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = torch.sqrt((channels_squared_sum / num_batches) - mean**2)
    return mean, std

# Assuming 'dataset' is your PyTorch dataset


def normalizer(dataset) -> Callable:
    """
    Creates the normalization of the dataset.
    
    Args:
        directory_path (str): The path to the CSV directory.
        current_configuration (str): The current configuration file.
    
    Returns:    
        normalize_transform (NormalizeTransform): The normalization transformer.
    """
    
    print(colored("Calculing normalization...", 'green'))
    mean, std = get_mean_std(dataset)
    print(colored(f"Normalization calculated: Mean={mean}, Std={std}", 'green'))
    
    # Creates the normalization transform
    def normalize_function(tensor):
        """
        Applies normalization to a single image.

        Args:
            image (Tensor): The input image tensor.

        Returns:
            Tensor: The normalized image tensor.
        """
        transform = transforms.Normalize(mean, std)
        return transform(tensor)
      
    return normalize_function
