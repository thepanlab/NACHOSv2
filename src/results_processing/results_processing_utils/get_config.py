import argparse
from contextlib import redirect_stderr
import io
import json
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
        '-j', '--json', '--load_json',
        help='Load settings from a JSON file.',
        required = False
    )
    
    # Tries to read the arguments
    try:
        shh = io.StringIO()
        
        with redirect_stderr(shh):
            args = parser.parse_args()
            
        if args.json is None:
            raise Exception(colored("You shouldn't see this...", 'red'))
        
        configuration_file = args.json
        
    
    except:
        
        configuration_file = prompt_json(default_config_file_name)
    
    
    # Opens the configuration file
    with open(configuration_file) as config:
        
        return json.load(config)



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
