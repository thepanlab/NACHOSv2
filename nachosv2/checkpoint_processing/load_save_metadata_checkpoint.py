import dill
import fasteners
import os
from typing import Optional, List
from pathlib import Path

from termcolor import colored
from nachosv2.checkpoint_processing.log_utils import get_rotation_dict, get_job_name, determine_use_lock, write_log_to_file


def read_log(config,
             test_fold: str,
             output_directory: str,
             rank: int = None,
             is_outer_loop: bool = False) -> Optional[dict]:
    """
    Reads a list of items in a log file.

    Args:
        log_directory (str): The directory containing the log files. In our case it's the output path of the training results.
        log_prefix (str): The prefix of the log. In our case it's the job's name.
        item_list_to_read (list of str): The list of dictionary keys to read.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
        process_rank (int): The process rank. Default is none. (Optional)

    Returns:
        extracted_items (dict): The dictionary of the specified items.
    """
    
    # Loads the log file
    log_filename = get_job_name(config, test_fold,
                                rank, is_outer_loop)

    log_filepath = Path(output_directory) / 'logs' / f"{log_filename}.dill"
    
    if log_filepath.exists() and log_filepath.stat().st_size > 0:
        with open(log_filepath, 'rb') as f:
            log_dict = dill.load(f)
    else:
        log_dict = None    
                          
    return log_dict


def write_log(config,
              indices_dict: dict,
              training_index: int,
              is_training_finished: bool,
              output_directory: str,
              rank: int = None,
              is_cv_loop: bool = False):
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
    
    log_dict = {
                "training_index": training_index,
                "test_fold": indices_dict["test"],
                "hp_config_index": indices_dict["hpo_configuration"]["hp_config_index"],
                "validation_fold": indices_dict["validation"],
                "is_training_finished": is_training_finished,
                }
    
    log_filename = get_job_name(config,
                                log_dict["test_fold"],
                                rank,
                                is_cv_loop)
    
    # use_lock = determine_use_lock(rank)

    write_log_to_file(output_directory,
                      log_filename,
                      log_dict)


def save_metadata_checkpoint(output_directory: str,
                             prefix_filename: str,
                             dict_to_save: dict,
                             use_lock: bool = True,
                             is_verbose_on: bool = False,
                             rank=Optional[int]):
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
        
    metadata_checkpoint_filepath = create_metadata_checkpoint_path(output_directory,
                                                                   prefix_filename,
                                                                   use_lock,
                                                                   rank)
    
    current_metadata_checkpoint = load_metadata_checkpoint(metadata_checkpoint_filepath,
                                                           is_verbose_on)
    
    if current_metadata_checkpoint is None:
        current_metadata_checkpoint = {}
    # Writes informations into
    with open(metadata_checkpoint_filepath, 'wb') as file_pointer:
        
        # Writes each element given in the data dictionary to the output dictionary 
        for key in dict_to_save:
            current_metadata_checkpoint[key] = dict_to_save[key]
        
        # Actually writes in the log file
        dill.dump(current_metadata_checkpoint, file_pointer)
        
        if is_verbose_on:  # If the verbose mode is activated
            print(colored("Metadata checkpoint successfully wrote.", 'cyan'))


def create_name_metadata_checkpoint(prefix_filename: str,
                                    is_verbose_on: bool = False,
                                    rank: Optional[int] = None) -> str:
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
    if rank is None:
        filename = f'{prefix_filename}.chk'    
    else:
        filename = f'{prefix_filename}_rank_{process_rank}.chk'

    if is_verbose_on:  # If the verbose mode is activated
        print(colored("Metadata checkpoint's filename created.", 'cyan'))
        
    return filename


def load_metadata_checkpoint(metadata_checkpoint_filepath: Path,
                             is_verbose_on: bool = False) -> Optional[dict]:
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
    
    # Checks if the log file exists
    if not metadata_checkpoint_filepath.exists() or \
        metadata_checkpoint_filepath.stat().st_size == 0:
            
        metadata_checkpoint = None
    else: # If it exists
        # Tries loading it as a dictionary
        try:
            with open(metadata_checkpoint_filepath, 'rb') as file_pointer:
                metadata_checkpoint = dill.load(file_pointer, encoding = 'latin1')
                if is_verbose_on: # If the verbose mode is activated
                    print(colored(f"Metada checkpoint from {metadata_checkpoint_filepath} successfully loaded.", 'cyan'))
        except:  # If the loading fails
            print(colored(f"Warning: Unable to open '{metadata_checkpoint_filepath}'", 'yellow'))
            metadata_checkpoint = None
    
    return metadata_checkpoint


def create_metadata_checkpoint_path(output_directory: str,
                                    prefix: str,
                                    use_lock: bool = True,
                                    rank=Optional[int]) -> Path:
    """
    Gets the logging path and the current log.

    Args:
        log_directory (str): The directory containing the log files. In our case it's the output path of the training results.
        log_prefix (str): The prefix of the log. In our case it's the job name.
        use_lock (bool): Whether to use a lock or not when writing results. Default is true. (Optional)
        process_rank (int): The process rank. Default is none. (Optional)
    """
    
    # Creates the logging path
    metadata_checkpoint_directory_path = Path(output_directory) / 'metadata_checkpoints' # The path to the metadata checkpoints
    
    # Checks if the output directory exists. If MPI, lock directory creation to avoid conflicts.
    if not metadata_checkpoint_directory_path.exists():
        if use_lock:  # = if MPI     
            lock_file_path = Path(output_directory) / 'metadata_checkpoint_lock.tmp'
            with fasteners.InterProcessLock(lock_file_path):  # Uses fasteners to lock directory creation if needed
                metadata_checkpoint_directory_path.mkdir(parents=True,
                                                         exist_ok=True)      
        else:
            metadata_checkpoint_directory_path.mkdir(parents=True,
                                                     exist_ok=True)
        
        
    # Checks if the log already exists and gets the path to write to.
    metadata_checkpoint_filename = create_name_metadata_checkpoint(prefix,
                                                                   rank) # Returns the formatted log file's path
    metadata_checkpoint_filepath = metadata_checkpoint_directory_path / metadata_checkpoint_filename
    
    return metadata_checkpoint_filepath


# def load_metadata_checkpoint(metadata_checkpoint_path: Path,
#                              is_verbose_on: bool=False):
#     """
#     Loads a log file from its path.

#     Args:
#         log_directory (str): The directory containing the log files. In our case it's the output path of the training results.
#         log_prefix (str): The prefix of the log. In our case it's the job name.
#         is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
#         process_rank (int): The process rank. Default is none. (Optional)

#     Returns:
#         unpickled_log (tuple): The tuple of the unpickled log. If no log, it will return None.
#     """   
    
#     # Checks if the log file exists
#     if not metadata_checkpoint_path.exists() or \
#     os.path.getsize(metadata_checkpoint_path) == 0:
#         metadata_checkpoint = None
#     else: # If it exists
#         # Tries loading it as a dictionary
#         try:
#             with open(metadata_checkpoint_path, 'rb') as file_pointer:
#                 metadata_checkpoint = dill.load(file_pointer, encoding = 'latin1')
                
#             if is_verbose_on: # If the verbose mode is activated
#                 print(colored(f"Checkpoint metadata in {log_file_path} successfully opened.", 'cyan'))     
#         except:  # If the loading fails
#             print(colored(f"Warning: Unable to open '{metadata_checkpoint_path}'", 'yellow'))
#             metadata_checkpoint = None
    
#     return metadata_checkpoint


def read_key_in_metadata_checkpoint(output_directory: str,
                                    prefix: str,
                                    keys_list_to_read: List[str],
                                    is_verbose_on: bool=False,
                                    process_rank: Optional[int]=None) -> dict:
    """
    Reads a list of items in a log file.

    Args:
        log_directory (str): The directory containing the log files. In our case it's the output path of the training results.
        log_prefix (str): The prefix of the log. In our case it's the job's name.
        item_list_to_read (list of str): The list of dictionary keys to read.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
        process_rank (int): The process rank. Default is none. (Optional)

    Returns:
        extracted_items (dict): The dictionary of the specified items.
    """
    
    metadata_checkpoint_path = create_metadata_checkpoint_path(output_directory,
                                                               prefix,
                                                               use_lock,
                                                               process_rank)
    
    metadata_checkpoint = load_metadata_checkpoint(metadata_checkpoint_path,
                                                   is_verbose_on)
    
    # If there is no log file, returns none
    if not metadata_checkpoint:
        extracted_items = None   
    
    # If there is a log file, gets the items within by key and returns them as a sub-dictionary
    else:
        extracted_items = {} # Creates the sub-dictionary that will be returned
        
        # For each asked key 
        for key in keys_list_to_read:
            if key in metadata_checkpoint: # If the current key exists, gets the data of the log file related to the current key
                extracted_items[key] = metadata_checkpoint[key]
                
                if is_verbose_on: # If the verbose mode is activated
                    print(colored(f"{key} extracted from Metadata Checkpoint.", 'cyan'))
            
    return extracted_items
