import os
from termcolor import colored


def get_all_from_directory(directory_path, return_full_path = False):
    """
    Creates a list of all files and subdirectories in the given directory.
    
    Agrs:
        directory_path (str): The path to the directory.
        TODO
    
    Returns:
        list (list of str): The list of paths to the files and subdirectories.
    """
    
    if _directory_checks(directory_path):

        list = _create_list(directory_path, get_files = True, get_subdirectories = True, return_full_path = return_full_path)
    
    else:
        list = []
    
    return list



def get_files_from_directory(directory_path, return_full_path = False):
    """
    Creates a list of all files in the given directory.
    
    Agrs:
        directory_path (str): The path to the directory.
    
    Returns:
        file_list (list of str): The list of paths to the files.
    """
    
    if _directory_checks(directory_path):

        file_list = _create_list(directory_path, get_files = True, get_subdirectories = False, return_full_path = return_full_path)
    
    else:
        file_list = []
        
    return file_list



def get_subdirectories_from_directory(directory_path, return_full_path = False):
    """
    Creates a list of all directories in the given directory.
    
    Agrs:
        directory_path (str): The path to the directory.
    
    Returns:
        subdirectories_list (list of str): The list of paths to the subdirectories.
    """
    
    if _directory_checks(directory_path):

        subdirectories_list = _create_list(directory_path, get_files = False, get_subdirectories = True, return_full_path = return_full_path)
    
    else:
        subdirectories_list = []
        
    return subdirectories_list



def _directory_checks(directory_path):
    """
    Checks if the given directory exists.
    
    Agrs:
        directory_path (str): The directory to check the existence.
    
    Returns:
        (bool): True if the given directory exists and is a directory.
    """
    
    if not os.path.exists(directory_path):
        print(colored(f"Warning: The given directory {directory_path} does not exist.", 'yellow'))
    
        return False
    
    elif not os.path.isdir(directory_path):
        print(colored(f"Warning: The given path {directory_path} is not a directory.", 'yellow'))
    
        return False
    
    else:
        return True



def _create_list(directory_path, get_files, get_subdirectories, return_full_path = True):
    """
    Creates the list to return.
    
    Agrs:
        directory_path (str): The directory to check the existence.
        get_files (bool): True to get the files.
        get_subdirectories (bool): True to get the subdirectories.
    
    Returns:
        returned_list (list of str): The list of what asked, subdirectories path or files path.
    """
    
    # Initializations
    returned_list = []
    subfiles = os.listdir(directory_path)
    
    
    # Sends warning if this is an empty directory
    if len(subfiles) == 0:
        print(colored(f"Warning: {directory_path} is an empty directory.", "yellow"))
    
    
    # For everything in the given directory
    for name in subfiles:
        
        # Creates the full path
        full_path = os.path.join(directory_path, name)
        
        
        # Checks if it needs to return the full path or not
        if return_full_path:
            name = full_path
        
        
        # If the mode is get subdirectories
        if get_subdirectories and os.path.isdir(full_path):
            returned_list.append(name)
        
        # If the mode is get files
        elif get_files and os.path.isfile(full_path):
            returned_list.append(name)
    
    
    return sorted(returned_list)
