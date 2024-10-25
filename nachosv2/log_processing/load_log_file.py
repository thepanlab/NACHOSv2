import dill
import os
from termcolor import colored

from src.log_processing.construct_log_file_path import construct_log_file_path


def load_log_file(log_directory, log_prefix, is_verbose_on = False, process_rank = None):
    """
    Loads a log file from its path.

    Args:
        log_directory (str): The directory containing the log files. In our case it's the output path of the training results.
        log_prefix (str): The prefix of the log. In our case it's the job name.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
        process_rank (int): The process rank. Default is none. (Optional)

    Returns:
        unpickled_log (tuple): The tuple of the unpickled log. If no log, it will return None.
    """
    
    # Creates the log file's path
    log_file_path = construct_log_file_path(log_directory, log_prefix, is_verbose_on, process_rank)
    
    
    # Checks if the log file exists
    if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
        unpickled_log = None
    
    
    else: # If it exists
        
        # Tries loading it as a dictionary
        try:
            with open(log_file_path, 'rb') as file_pointer:
                unpickled_log = dill.load(file_pointer, encoding = 'latin1')
                
                if is_verbose_on: # If the verbose mode is activated
                    print(colored(f"Unpickled log from {log_file_path} successfully loaded.", 'cyan'))
            
        except: # If the loading fails
            print(colored(f"Warning: Unable to open '{log_file_path}'", 'yellow'))
            unpickled_log = None
    
    
    return unpickled_log