from termcolor import colored

from src.log_processing.load_log_file import load_log_file


def read_item_list_in_log(log_directory, log_prefix, item_list_to_read, is_verbose_on = False, process_rank = None):
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
    log_data = load_log_file(log_directory, log_prefix, process_rank)
    
    
    # If there is no log file, returns none
    if not log_data:
        extracted_items = None
    
    
    # If there is a log file, gets the items within by key and returns them as a sub-dictionary
    else:
        extracted_items = {} # Creates the sub-dictionary that will b returned
        
        # For each asked key 
        for key in item_list_to_read:
            
            if key in log_data : # If the current key exists, gets the data of the log file related to the current key
                extracted_items[key] = log_data[key]
                
                if is_verbose_on: # If the verbose mode is activated
                    print(colored(f"Data load for the subject {key}.", 'cyan'))
                    
            
    return extracted_items
