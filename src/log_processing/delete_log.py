import os

from src.log_processing.construct_log_file_path import construct_log_file_path
        
        
def delete_log_file(log_directory, log_prefix, process_rank = None):
    """
    Deletes a log file.

    Args:
        log_directory (str): The directory containing the log files. In our case it's the output path of the training results.
        log_prefix (str): The prefix of the log. In our case it's the job name
        process_rank (int): The process rank. Default is none. (Optional)
    """    
    
    log_file_path  = construct_log_file_path(log_directory, log_prefix, process_rank)
    
    # Checks if the log file exists, remove if it does
    if os.path.exists(log_file_path ):
        os.remove(log_file_path)  
        