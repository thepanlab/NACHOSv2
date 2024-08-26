import argparse
import json
import os
from termcolor import colored


def parse_mpi_training_configs(default_config_directory_name):
    """
    Parse the configurations from the command line.
    
    Args:
        default_config_directory_name (str): The default configuration location.
    """
    
    # Command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file', '--config_file',
        type=str, default=None, required=False,
        help='Load settings from a JSON file.'
    )
    
    parser.add_argument(
        '--folder', '--config_folder',
        type=str, default=default_config_directory_name, required=False,
        help='Load settings from a JSON file.'
    )
    
    args = parser.parse_known_args()
     
    
    # Check for command line errors
    if not (args[0].file or args[0].folder):
        raise Exception("Error: no configuration file or folder specified.")
    if (args[0].file and args[0].folder) and args[0].folder != default_config_directory_name:
        raise Exception(colored("Error: Please only specify a configuration file or directory, not both.", 'red'))

    # Read in a single file
    if args[0].file:
        if not os.path.exists(args[0].file):
            raise Exception(colored(f"Error: The file '{args[0].file}' does not exist."))
        with open(args[0].file) as fp:
            return [json.load(fp)]
            
    # Read in a directory of files
    elif args[0].folder:
        if not os.path.isdir(args[0].folder):
            raise Exception(colored(f"Error: The directory '{args[0].folder}' does not exist."))
        
        # Read in each valid config file within
        configs = []
        for file in os.listdir(args[0].folder):
            full_path = os.path.join(args[0].folder, file)
            if not os.path.isdir(full_path) and full_path.endswith(".json"):
                with open(full_path) as fp:
                    configs.append(json.load(fp))
        return configs
    
    # Shouldn't reach here
    else:
        raise Exception(colored("Error: Unknown error reached in the configuration parsing.", 'red'))
    