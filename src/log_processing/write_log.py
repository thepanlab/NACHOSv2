import dill
import fasteners
import os
from termcolor import colored

from src.log_processing.construct_log_file_path import construct_log_file_path
from src.log_processing.load_log_file import load_log_file
from src.log_processing.log_utils import *


def write_log(config, testing_subject, rotation, log_directory, log_rotations = None, validation_subject = None, rank = None, is_outer_loop = False):
    """
    Writes a log entry to a file with information about the current rotation and job.

    Args:
        config (dict): Configuration dictionary containing job name information.
        testing_subject (str): The identifier for the testing subject.
        rotation (int): The current rotation value to be recorded.
        log_directory (str): The directory where the log file will be written.
        
        log_rotations (dict, optional): Dictionary containing previous rotation logs. Default is None.
        validation_subject (str, optional): The identifier for the validation subject. Default is None.
        rank (int, optional): The rank or process identifier. Default is None.
        is_outer_loop (bool, optional): Indicates if this is the outer loop. Default is False.
    """
    
    rotation_dict = get_rotation_dict(testing_subject, rotation, log_rotations)
    
    job_name = get_job_name(config, testing_subject, validation_subject, rank, is_outer_loop)
    
    use_lock = determine_use_lock(rank)

    write_log_to_file(log_directory, job_name, {'current_rotation': rotation_dict}, use_lock, is_verbose_on = False, process_rank = None)
    
    
        
def write_log_to_file(log_directory, log_prefix, data_dict, use_lock = True, is_verbose_on = False, process_rank = None):
    """
    Writes a log of the state to file. Can add individual items.

    Args:
        log_directory (str): The directory containing the log files. In our case it's the output path of the training results.
        log_prefix (str): The prefix of the log. In our case it's the job name.
        data_dict (dict): A dictionary of the relevant status info.
        
        use_lock (bool): Whether to use a lock or not when writing results. Default is true. (Optional)
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
        process_rank (int): The process rank. Default is none. (Optional)
    """
    
    # Gets the log's path and value
    log_path, current_log = _writing_log_preparation(log_directory, log_prefix, use_lock, process_rank)
    
    
    # Writes informations into
    with open(log_path, 'wb') as file_pointer:
        
        # Creates the output dictionary if it doesn't exists
        if current_log == None:
            current_log = {}
        
        
        # Writes each element given in the data dictionary to the output dictionary 
        for key in data_dict:
            current_log[key] = data_dict[key]
        
        
        # Actually writes in the log file
        dill.dump(current_log, file_pointer)
        
        
        if is_verbose_on: # If the verbose mode is activated
            print(colored("Log successfully writes.", 'cyan'))



def _writing_log_preparation(log_directory, log_prefix, use_lock = True, process_rank = None):
    """
    Gets the logging path and the current log.

    Args:
        log_directory (str): The directory containing the log files. In our case it's the output path of the training results.
        log_prefix (str): The prefix of the log. In our case it's the job name.
        use_lock (bool): Whether to use a lock or not when writing results. Default is true. (Optional)
        process_rank (int): The process rank. Default is none. (Optional)
    """
    
    # Creates the logging path
    logging_path = os.path.join(log_directory, 'logging')
    
    # Checks if the output directory exists. If MPI, lock directory creation to avoid conflicts.
    if not os.path.exists(logging_path):
        
        if use_lock: # = if MPI
            with fasteners.InterProcessLock(os.path.join(log_directory, 'logging_lock.tmp')): # Uses fasteners to lock directory creation if needed
                os.makedirs(logging_path)
                
        else:
            os.makedirs(logging_path) # Creates the directory
        
        
    # Checks if the log already exists and gets the path to write to.
    log_path = construct_log_file_path(log_directory, log_prefix, process_rank) # Returns the formatted log file's path
    current_log = load_log_file(log_directory, log_prefix, process_rank) # Returns the tuple of the unpickled log, or none
    
    
    return log_path, current_log
