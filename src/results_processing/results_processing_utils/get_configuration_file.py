import argparse
import json
import sys
from termcolor import colored


def parse_json(default_config_file_name):
    """
    Parses the config file from the command line.
    
    Args:
        default_config_file_name (str): The default configuration file.
    """
    
    # Parses the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--f', '--file', '--config_file',
        dest = 'json_file',
        help = 'Load settings from a JSON file.',
        required = False,
        default = None
    )
    
    args = parser.parse_args()
    
    # Tries to read the arguments
    try:
        
        # If a file is given, uses it
        if args.json_file:
            
            configuration_file = args.json_file
            
        
        # Else, ask for a file
        else:
            
            configuration_file = prompt_json(default_config_file_name)
        
        
        # Opens the configuration file
        with open(configuration_file) as config:
            
            return json.load(config)
    
    
    # Desl ith the exceptions
    except argparse.ArgumentError:
        print(colored("Error: Invalid argument provided.", 'red'))
        sys.exit(1)
        
    except FileNotFoundError:
        print(colored(f"Error: Configuration file '{configuration_file}' not found.", 'red'))
        sys.exit(1)
        
    except json.JSONDecodeError:
        print(colored(f"Error: '{configuration_file}' is not a valid JSON file.", 'red'))
        sys.exit(1)
        
    except Exception as e:
        print(colored(f"An unexpected error occurred: {str(e)}", 'red'))
        sys.exit(1)



def prompt_json(default_config_file_name):
    """
    Prompts the user for a config name. 
    
    Args:
        default_config_file_name (str): The default configuration location.
    """
    
    # Prompts for input
    config_file = input(colored("Please enter the config path, or press enter to use the default path:\n", 'cyan'))
    
    # If no input, use a default file. Load in the configuration json.
    if not config_file:
        config_file = default_config_file_name
    
    
    return config_file
