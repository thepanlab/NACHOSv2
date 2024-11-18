def get_job_name(config, testing_subject, validation_subject = None, rank = None, is_outer_loop = False):
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
    
    if rank:
        if is_outer_loop:
            job_name = f"{config['job_name']}_test_{testing_subject}"
            
        else:
            job_name = f"{config['job_name']}_test_{testing_subject}_val_{validation_subject}"
            
    else:
        job_name = config['job_name']
        
        
    return job_name


def get_rotation_dict(testing_subject, rotation, log_rotations = None):
    """
    Updates the rotation dictionary with the current rotation for the testing subject.

    Args:
        testing_subject (str): The identifier for the testing subject.
        rotation (int): The current rotation value to be recorded.
        log_rotations (dict, optional): Dictionary containing previous rotation logs. Default is None.

    Returns:
        rotation_dict (dict): The updated rotation dictionary.
    """
    
    if not log_rotations:
            rotation_dict = {testing_subject: rotation + 1}        
    else:
        rotation_dict = log_rotations['current_rotation']
        rotation_dict[testing_subject] = rotation + 1
            
    return rotation_dict


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
