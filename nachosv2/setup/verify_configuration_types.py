import sys
from termcolor import colored


def verify_configuration_types(config_file):
    """
    Checks that the arguments in the configuration file are of the right type.
    
    Args:
        configuration_file (JSON): The configuration file.
        configuration_path (str): The path of the JSON config file.
    """

    verify_functions = [
        _verify_string,
        _verify_boolean,
        _verify_number,
        _verify_list,
        _verify_dictionary
        ]

    # For each argument
    for key, value in config_file.items():
        for verify_function in verify_functions:
            verify_function(config_file, key, value)
                          

def _print_error(config_file, key, expected_type):
    """
    Prints an error message and exits the program.

    Args:
        config_path (str): Path to the configuration file.
        key (str): Argument key.
        expected_type (str): Expected type description.
    Raises:
        ValueError: Always.
    """
    error_message = f"The '{key}' argument in '{config_file}' must be a {expected_type}."
    raise ValueError(error_message)


def _verify_string(config_file, key, value):
    """
    Verifies the correct type of an expected string argument value
    
    Args:
        key (str): The argument key.
        value (any): Argument value.
        
    Raises:
        ValueError: If the argument is not a string.
    """
    
    string_keys = {'data_input_directory', 'output_path', 'job_name', 'selected_model_name'}
    
    if key in string_keys:
        if not isinstance(value, str):
            _print_error(config_file, key, "string")


def _verify_boolean(config_file, key, value):
    """
    Verifies the correct type of an expected boolean argument value
    
    Args:
        key (str): Argument key.
        argument (various): The argument linked to the key.
        
    Raises:
        ValueError: When the argument is not a boolean.
    """
    
    boolean_keys = {'shuffle_the_images', 'shuffle_the_folds', 'do_cropping', 'bool_nesterov'}
    
    if key in boolean_keys:
        if not isinstance(value, bool):
            _print_error(config_file, key, "boolean")
            

def _verify_number(config_file, key, value):
    """
    Verifies the correct type of an expected int/float argument value

    Args:
        config_path (str): Path to the configuration file.
        key (str): Argument key.
        value (any): Argument value.

    Raises:
        ValueError: If the argument is not an integer or float.
    """
    
    number_keys = {'batch_size', 'channels', 'decay', 'epochs', 'learning_rate', 'momentum', 
                   'patience', 'k_epoch_checkpoint_frequency', 'seed', 'target_height', 'target_width'}
    if key in number_keys:
        if not isinstance(value, (int, float)):
            _print_error(config_file, key, "number (int or float)")
            

def _verify_list(config_file, key, value):
    """
    Verifies the correct type of an expected list argument value as well as
    the type of its elements.

    Args:
        config_path (str): Path to the configuration file.
        key (str): Argument key.
        value (any): Argument value.

    Raises:
        ValueError: If the argument is not a list or list elements are of incorrect type.
    """
    
    list_keys = {
        'cropping_position': int,
        'image_size': int,
        'class_names': str,
        'subject_list': str,
        'test_subjects': str,
        'validation_subjects': str,
        'metrics': str
        }
    
    if key in list_keys:
        if not isinstance(value, list):
            _print_error(config_file, key, "list")
        
        # Check list elements type if specified
        elem_type = list_keys[key]
        if elem_type:
            for item in value:
                if not isinstance(item, elem_type):
                    _print_error(config_file, key, f"list of {elem_type.__name__}")


def _verify_dictionary(config_file, key, value):
    """
    Verifies the correct type of an expected dict argument value as well as
    the type of its elements.
    
    Args:
        config_path (str): Path to the configuration file.
        key (str): Argument key.
        value (any): Argument value.

    Raises:
        ValueError: If the argument is not a dictionary.
    """
    
    verify_functions_inside_dictionary = [
        _verify_string,
        _verify_boolean,
        _verify_number,
        _verify_list
        ]
    
    if key == 'hyperparameters' and isinstance(value, dict):
        for sub_key, sub_value in value.items():
            for verify_function in verify_functions_inside_dictionary:
                verify_function(config_file, sub_key, sub_value)