import sys
from termcolor import colored


def check_configuration_types(configuration_file, configuration_path):
    """
    Checks that the arguments in the configuration file are of the right type.
    
    Args:
        configuration_file (JSON): The configuration file.
        configuration_path (str): The path of the JSON config file.
    """

    # For each argument
    for key, argument in configuration_file.items():
        
        _check_string(configuration_path, key, argument)
        
        _check_boolean(configuration_path, key, argument)
                          
        _check_number(configuration_path, key, argument)
            
        _check_list(configuration_path, key, argument)
        
        _check_dictionary(configuration_path, key, argument)
            


def _check_string(configuration_path, key, argument):
    """
    Checks that the arguments that should be strings are strings.
    
    Args:
        key (str): The argument key.
        argument (various): The argument linked to the key.
        
    Raises:
        Exception: When the argument is not a string.
    """
    
    if (key == 'data_input_directory'
        or key == 'output_path'
        or key == 'job_name'
        or key == 'selected_model_name'):
        
        # Checks if it's a string
        try:
            assert isinstance(argument, str), f"The '{key}' argument in '{configuration_path}' must be a string."
            
        except AssertionError as error:
            print(colored(error, 'red'))
            sys.exit()



def _check_boolean(configuration_path, key, argument):
    """
    Checks that the arguments that should be booleans are booleans.
    
    Args:
        key (str): The argument key.
        argument (various): The argument linked to the key.
        
    Raises:
        Exception: When the argument is not a boolean.
    """
    
    if (key == 'shuffle_the_images'
        or key == 'shuffle_the_folds'
        or key == 'do_cropping'
        or key == 'bool_nesterov'):
        
        # Checks if it's a boolean
        try:
            assert isinstance(argument, bool), f"The '{key}' argument in '{configuration_path}' must be a boolean."
            
        except AssertionError as error:
            print(colored(error, 'red'))
            sys.exit()
            


def _check_number(configuration_path, key, argument):
    """
    Checks that the arguments that should be numbers are numbers.
    
    Args:
        key (str): The argument key.
        argument (various): The argument linked to the key.
        
    Raises:
        Exception: When the argument is not a int or a float.
    """
    
    if (key == 'batch_size'
        or key == 'channels'
        or key == 'decay'
        or key == 'epochs'
        or key == 'learning_rate'
        or key == 'momentum'
        or key == 'patience'
        or key == 'k_epoch_checkpoint_frequency'
        or key == 'seed'
        or key == 'target_height'
        or key == 'target_width'):
        
        # Checks if it's an int or a float
        try:
            assert isinstance(argument, (int, float)), f"The '{key}' argument in '{configuration_path}' must be an integer or a float."
            
        except AssertionError as error:
            print(colored(error, 'red'))
            sys.exit()
            


def _check_list(configuration_path, key, argument):
    """
    Checks that the arguments that should be lists are lists.
    
    Args:
        key (str): The argument key.
        argument (various): The argument linked to the key.
        
    Raises:
        Exception: When the argument is not a list or when the arguments in the list are not int or string.
    """
    
    if (key == 'cropping_position'
        or key == 'class_names'
        or key == 'subject_list'
        or key == 'test_subjects'
        or key == 'validation_subjects'
        or key == 'image_size'
        or key == 'metrics'):
        
        
        # Checks if it's a list
        try:
            assert isinstance(argument, list), f"The '{key}' argument in '{configuration_path}' must be a list."
            
            
            # Checks if it's a list of int
            if (key == 'cropping_position'
                or key == 'image_size'):
                
                for item in argument:
                    assert isinstance(item, int), f"The elements of the '{key}' in '{configuration_path}' list must be integers."
            
            
            # Checks if it's a list of strings
            elif (key == 'class_names'
                or key == 'subject_list'
                or key == 'test_subjects'
                or key == 'validation_subjects'):
                
                for item in argument:
                    assert isinstance(item, str), f"The elements of the '{key}' in '{configuration_path}' list must be string."

            
        except AssertionError as error:
            print(colored(error, 'red'))
            sys.exit()



def _check_dictionary(configuration_path, key, argument):
    """
    Checks that the arguments that should be lists are lists.
    
    Args:
        key (str): The argument key.
        argument (various): The argument linked to the key.
        
    Raises:
        Exception: When the argument is not a string.
    """
    
    if key == 'hyperparameters':
        
        # For each argument
        for sub_key, sub_argument in argument.items():
            
            _check_string(configuration_path, sub_key, sub_argument)
            
            _check_boolean(configuration_path, sub_key, sub_argument)
                            
            _check_number(configuration_path, sub_key, sub_argument)
                
            _check_list(configuration_path, sub_key, sub_argument)
            