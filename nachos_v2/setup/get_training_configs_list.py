import json
import os
from termcolor import colored

from nachos_v2.setup.command_line_parser import DEFAULT_CONFIGURATION_PATH
    
    
def get_training_configs_list(configuration_file_path, configuration_folder_path):
    """
    Gets the configuration file(s), whatever it's one file or a folder.
    
    Args:
        configuration_file_path (str): The path of the configuration file.
        configuration_folder_path (str): The path of the configuration folder.
        
    Returns:
        list_of_configs (list of JSON): The list of configuration file.
        list_of_configs_paths (list of str): The list of configuration file's paths.
    """
    
    list_of_configs = [] # The list of config file
    list_of_configs_paths = [] # The list of config file's paths
    
    
    # Checks for command line errors
    if not (configuration_file_path or configuration_folder_path): # Shouldn't be used because a default configuration folder path exists
        raise Exception(colored("Error: no configuration file or folder specified.", 'red'))
    
    if (configuration_file_path and configuration_folder_path) and configuration_folder_path != DEFAULT_CONFIGURATION_PATH:
        raise Exception(colored("Error: Please only specify a configuration file or directory, not both.", 'red'))


    # If a file is specified, reads it and returns it
    if configuration_file_path:
        
        if not os.path.exists(configuration_file_path): # Checks if the file exists
            raise Exception(colored(f"Error: The file '{configuration_file_path}' does not exist.", 'red'))
        
        with open(configuration_file_path) as file_pointer: # Guarantees that the file will be close, even if there is an reading error
            list_of_configs.append(json.load(file_pointer)) # Add the file's path to the list that will be returned
            list_of_configs_paths.append(configuration_file_path)
    
    
    # If a folder is specified, reads all the files in it and returns a list of data
    # If neither file or folder is specified, should go here with the default configuration folder path
    else:
        
        if not os.path.isdir(configuration_folder_path): # Checks if the folder exists
            raise Exception(colored(f"Error: The directory '{configuration_folder_path}' does not exist.", 'red'))
        
        # Reads in each valid config file within the folder
        for file in os.listdir(configuration_folder_path):
            full_file_path = os.path.join(configuration_folder_path, file) # Gets the full path of the current file
            
            if not os.path.isdir(full_file_path) and full_file_path.endswith(".json"): # If the full path is a .json file
                with open(full_file_path) as file_pointer: # Guarantees that the file will be close, even if there is an reading error
                    list_of_configs.append(json.load(file_pointer)) # Adds the current file's path to the list that will be returned
                    list_of_configs_paths.append(full_file_path)
                    
                    
    return list_of_configs, list_of_configs_paths
    