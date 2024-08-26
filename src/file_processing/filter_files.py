import os
import regex as re


def filter_files_by_extension(file_list, subjects_list, extension):
    """
    Filters a list of files and directories, returning only the files of the given extension whose base names are in the subjects list.

    Args:
        file_list (list of str): The list of files and directories to filter.
        subjects_list (list of str): The list of subject names to keep.
        extension (str): The extension of the files you want to keep.

    Returns:
        filtered_files (list of str): The list of files of the given extension whose base names are in the subjects list.
    """
    
    # Initialization
    filtered_files = []


    # Iterates over each item in the file_list
    for file in file_list:
        
        # Checks if the item is a file with the correct extension
        if file.endswith(extension):
            
            # Extracts the base name of the file (without the directory path and extension)
            base_name = os.path.splitext(os.path.basename(file))[0]
            
            # Keeps the file if the base name is in the subjects list
            if base_name in subjects_list:
                filtered_files.append(file)


    return filtered_files



def filter_files_by_type(files, is_validation, is_testing, is_outer):
    """
    Filters files based on validation/testing criteria.
    
    TODO
    """
    
    if is_outer:
        return [f for f in files if re.search('.*_test_.*', f.split('/')[-1])]
    
    if is_validation:
        return [f for f in files if re.search('_test_.*_val_.*_val', f.split('/')[-1])]
    
    elif is_testing:
        return [f for f in files if re.search('_test_.*_val_.*_test_', f.split('/')[-1])]
    
    return files



def filter_files_for_predictions(files):
    """
    Filter a list of CSV files to keep only those ending with "_predicted_labels.csv"

    Args:
    file_list (list): A list of file names (strings)

    Returns:
    list: A list of file names that end with "_predicted_labels.csv"
    """
    
    # Initialization
    filtered_files = []


    # Iterates through each file in the input list
    for file in files:
        
        # Checks if the file name ends with "_predicted_labels.csv"
        if file.endswith("_predicted_labels.csv"):
            
            filtered_files.append(file)


    return filtered_files



def filter_files_by_index(files, is_index):
    """
    Filters files based on index criteria.
    
    TODO
    """
    
    if is_index is None:
        return files
    
    return [f for f in files if ("index" in f) == is_index]