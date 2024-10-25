import os
from termcolor import colored


def construct_log_file_path(log_directory, log_prefix, is_verbose_on = False, process_rank = None):
    """
    Gets a log name for the given conditions.

    Args:
        log_directory (str): The directory containing the log files. In our case it's the output path of the training results.
        log_prefix (str): The prefix of the log. In our case it's the job name.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
        process_rank (int): The process rank. Default is none. (Optional)

    Returns:
        log_filepath (str): The formatted log file's path.
    """
    
    # Appends the rank to the job name if given for the log
    if process_rank is None:
        log_filepath = os.path.join(log_directory, 'logging', f'{log_prefix}.log')
    
    else:
        log_filepath = os.path.join(log_directory, 'logging', f'{log_prefix}_rank_{process_rank}.log')


    if is_verbose_on: # If the verbose mode is activated
        print(colored("Log file's path created.", 'cyan'))
        

    return log_filepath
