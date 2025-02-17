from pathlib import Path
import dill

def get_job_name(config: dict,
                 test_fold: str,
                 rank: int = None,
                 is_cv_loop: bool = False):
    """
    Generates a job name based on the provided configuration, testing, and validation subjects.

    Args:
        config (dict): Configuration dictionary containing job name information.
        testing_subject (str): The identifier for the testing subject.
        validation_subject (str, optional): The identifier for the validation subject. Default is None.

        rank (int, optional): The rank or process identifier. Default is None.
        is_outer_loop (bool, optional): Indicates if this is the outer loop. Default is False.

    Returns:
        job_name (str): The generated job name.
    """
    
    # if rank:
    #     if is_outer_loop:
    #         job_name = f"{config['job_name']}_test_{test_fold}" 
    #     else:
    #         job_name = f"{config['job_name']}_test_{test_fold}" + \
    #                    f"_val_{validation_fold}"
    # else:
    #     job_name = config['job_name']     

    job_name = config['job_name']     
        
    return job_name


def get_rotation_dict(test_fold: str,
                      rotation_index: int):
    """
    Updates the rotation dictionary with the current rotation for the testing subject.

    Args:
        testing_subject (str): The identifier for the testing subject.
        rotation (int): The current rotation value to be recorded.
        log_rotations (dict, optional): Dictionary containing previous rotation logs. Default is None.

    Returns:
        rotation_dict (dict): The updated rotation dictionary.
    """

    return {test_fold: rotation_index}


def determine_use_lock(rank):
    """
    Determines the value of use_lock based on whether rank is not None.

    Args:
        rank (Any): The value to test, can be of any type.

    Returns:
        use_lock (bool): True if rank is not None, otherwise False.
    """

    if rank is not None:
        use_lock = True
    else:
        use_lock = False

    return use_lock


def write_log_to_file(log_folder: str,
                      log_filename: str,
                      dict_to_save: dict):
    """_summary_

    Args:
        log_folder (str): folder to store the log file
        log_filename (str): filename of log file
        dict_to_save (dict): dictionary to be saved
    """
    
    folder_logs_path = Path(log_folder) / 'logs'
    folder_logs_path.mkdir(parents=True, exist_ok=True)
    filepath = folder_logs_path / f"{log_filename}.dill"
    
    # Open a file in write-binary mode
    with open(filepath, 'wb') as file:
        # Use dill to dump the dictionary into the file
        dill.dump(dict_to_save, file)
    