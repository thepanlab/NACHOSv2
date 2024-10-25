from termcolor import colored

from src.data_processing.calculate_mean_std import calculate_mean_std
from src.file_processing.filter_files import filter_files_by_extension
from src.data_processing.full_dataset import FullDataset
from src.file_processing.get_from_directory import get_files_from_directory
from src.data_processing.normalize_transform import NormalizeTransform


def normalize(directory_path, current_configuration):
    """
    Creates the normalization of the dataset.
    
    Args:
        directory_path (str): The path to the CSV directory.
        current_configuration (str): The current configuration file.
    
    Returns:    
        normalize_transform (NormalizeTransform): The normalization transformer.
    """
    
    print(colored("Calculing normalization...", 'green'))
    
    
    # Gets the list of files
    file_list = get_files_from_directory(directory_path)
    

    # Gets only the wanted CSV files
    csv_file_list = filter_files_by_extension(file_list, current_configuration['subject_list'] , ".csv")


    # Creates the dataset containing all of the data
    full_dataset = FullDataset(current_configuration['data_input_directory'], directory_path, csv_file_list)
    
    
    # Calculates the mean and the standard deviation
    mean, std = calculate_mean_std(full_dataset)
    
    
    # Creates the normalization transform
    normalize_transform = NormalizeTransform(mean, std)
    
    
    # Deletes the dataset
    del full_dataset
    
    
    print(colored("Normalization calculed.", 'green'))
    
    
    return normalize_transform
