import argparse
from termcolor import colored


# Definition of default arguments
DEFAULT_CONFIGURATION_PATH = 'scripts/config_files'
DEFAULT_VERBOSE = False
DEFAULT_EXECUTION_DEVICE = "cuda:1"


def add_arguments_to_parser(parser):
    """
    Given an existing parActually parses the arguments.
    
    Agrs:
        parser (ArgumentParser): The parser.
    """
    
    # Definition of all arguments
    parser.add_argument( # Allows to specify the config file in command line
        '--file', '--config_file',
        type = str, default = None, required = False,
        help = 'Load settings from a JSON file.'
    )
    
    parser.add_argument( # Allows to specify the config folder in command line
        '--folder', '--config_folder',
        type = str, default = DEFAULT_CONFIGURATION_PATH, required = False,
        help = 'Load settings from a JSON folder.'
    )
    
    parser.add_argument( # Allows to activate verbose mode in command line
        '--verbose', '--v',
        action = 'store_true', default = DEFAULT_VERBOSE, required = False,
        help = 'Activate verbose mode if specified'
    )
    
    parser.add_argument( # Allows to specify the execution device in command line
        '--device', '--d',
        type = str, default = DEFAULT_EXECUTION_DEVICE, required = False,
        help = 'Change the execution device'
    )
    
    
def parse_command_line_args():
    """
    Parses command-line arguments.
    
    Returns:
        recognized_arguments (dict): The dictionnary of all recognized arguments
            and default ones for those not specified
    """
    
    # Instantiates the parser object to add it new possible arguments
    parser = argparse.ArgumentParser()
    
    # Actually parses the arguments
    add_arguments_to_parser(parser)
    
    # Splits the recognized arguments from the unrecognized ones
    recognized_arguments, unrecognized_arguments = parser.parse_known_args()
        # recognized_arguments is a namespace of the recognized arguments
        # unrecognized_arguments is a list of the unrecognized arguments

    # Converts the namespace into a dictionary for easier use
    recognized_arguments = vars(recognized_arguments)
    
    # Warns for every unrecognized arguments
    if unrecognized_arguments:
        print(colored(f"Warning: unrecognized argument(s): {unrecognized_arguments}.", 'yellow'))
    
    # Prints if verbose mode is activated
    if recognized_arguments['verbose']:
        print(colored("Verbose mode activated.", 'cyan'))
    
    # Returns the recognized arguments
    return recognized_arguments
