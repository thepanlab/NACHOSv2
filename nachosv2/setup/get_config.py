import os
from pathlib import Path
from typing import List
from termcolor import colored

import yaml

from nachosv2.setup.command_line_parser import DEFAULT_CONFIGURATION_PATH
from nachosv2.setup.verify_configuration_types import verify_configuration_types


def get_config(config_file_path: str) -> dict:
    """
    Gets the configuration file(s), whatever it's one file or a folder.
    
    Args:
        configuration_file_path (str): The path of the configuration file.
        configuration_folder_path (str): The path of the configuration folder.
        
    Returns:
        list_of_configs (list of JSON): The list of configuration file.
        list_of_configs_paths (list of str): The list of configuration file's paths.
    """
       
    # Checks for command line errors
    # Shouldn't be used because a default configuration folder path exists
    if not config_file_path:
        raise ValueError(colored("No configuration file or folder specified.",'red'))

    # If a file is specified, reads it and returns it
    if config_file_path:
        if not Path(config_file_path).exists(): # Checks if the file exists
            raise FileNotFoundError(colored(f"Error: The file '{config_file_path}' does not exist.", 'red'))
        
        with open(config_file_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)            
                    
    return config_dict


def get_config_list(config_file_path: str,
                    config_folder_path: str) -> List[dict]:
    """
    Gets the configuration file(s), whatever it's one file or a folder.
    
    Args:
        configuration_file_path (str): The path of the configuration file.
        configuration_folder_path (str): The path of the configuration folder.
        
    Returns:
        list_of_configs (list of JSON): The list of configuration file.
        list_of_configs_paths (list of str): The list of configuration file's paths.
    """
    
    config_list = [] # The list of config file
    
    # Checks for command line errors
    # Shouldn't be used because a default configuration folder path exists
    if not (config_file_path or config_folder_path):
        raise ValueError(colored("No configuration file or folder specified.",'red'))
    
    if config_file_path and config_folder_path != DEFAULT_CONFIGURATION_PATH:
        raise ValueError(colored("Please ONLY specify a configuration file or directory, not both.", 'red'))


    # If a file is specified, reads it and returns it
    if config_file_path:
        if not Path(config_file_path).exists(): # Checks if the file exists
            raise Exception(colored(f"Error: The file '{config_file_path}' does not exist.", 'red'))
               
        # with open(config_file_path) as file:  # Guarantees that the file will be close, even if there is an reading error
        #     dict_config = json.load(file)
        #     verify_configuration_types(dict_config)
        with open(config_file_path, 'r') as f:
            dict_config = yaml.load(f, Loader=yaml.SafeLoader)            
            config_list.append(dict_config)  # Add the file's path to the list that will be returned
    
    # If a folder is specified, reads all the files in it and returns a list of data
    # If neither file or folder is specified, should go here with the default configuration folder path
    else:
        
        if not Path(config_folder_path).is_dir():  # Checks if the folder exists
            raise ValueError(colored(f"Error: The directory '{config_file_path}' does not exist.", 'red'))
        
        # Reads in each valid config file within the folder
        for filename in Path(config_folder_path).glob("*.yml"):
            full_file_path = os.path.join(config_folder_path, filename) # Gets the full path of the current file
            
            if not os.path.isdir(full_file_path) and full_file_path.endswith(".yml"): # If the full path is a .json file
                with open(full_file_path, 'r') as f:
                    dict_config = yaml.load(f, Loader=yaml.SafeLoader)            
                    config_list.append(dict_config)  # Add the file's path to the list that will be returned
                    
    return config_list
    