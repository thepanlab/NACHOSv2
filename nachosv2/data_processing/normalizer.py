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
