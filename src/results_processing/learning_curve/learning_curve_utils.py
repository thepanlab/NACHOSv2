import regex as re
from termcolor import colored


def get_subject_name(file_name):
    """
    Gets the subject id

    Args:
        file_name (str): Name of the input file.

    Raises:
        Exception: When the file name has an incorrect format.

    Returns:
        str: The subject's name.
    """
    
    try:
        subject_search = re.search('_test_.*_val_history', file_name)
        subject_name = subject_search.captures()[0].split("_")[2]
        
        return subject_name
    
    except:
        
        try:
            subject_search = re.search('_test_.*_history', file_name)
            subject_name = subject_search.captures()[0].split("_")[2]
            
            return subject_name
        
        except:
            raise Exception(colored(f"Error: File name doesn't contain '_test_*_val_history' or '*_test_*_history format: \n\t{file_name}", 'red'))



def file_verification(files_list):
    """
    Verifies files are valid.

    Args:
        files_list (list): A list of file names

    Returns:
        list: A list of good file names.
    """
    
    verified_list = []
    
    for file in files_list:
        history_check = re.search("history", file)
        
        if history_check != None:
            verified_list.append(file)
            
        else:
            pass
        
    return verified_list
