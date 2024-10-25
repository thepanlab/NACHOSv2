import os
import sys
from termcolor import colored

from src.file_processing.filter_files import filter_files_by_index, filter_files_by_type, filter_files_for_predictions
from src.file_processing.get_from_directory import get_all_from_directory, get_files_from_directory, get_subdirectories_from_directory
# from src.results_processing.results_processing_utils.predicted_formatter.predicted_formatter import main as reformat


sys.setrecursionlimit(10**6)

# TODO

def is_outer_loop(data_path):
    """
    Checks if a given data path is of an inner or outer loop.

    Args:
        data_path (str): The path to a data directory.

    Raises:
        Exception: If a data path has no files.

    Returns:
        bool: If a directory is of an outer-loop structure.
    """
    # Get the test-fold directories
    test_subject_paths = get_all_from_directory(data_path, return_full_path = True)
    if not test_subject_paths:
        raise Exception(colored(f'Error: no files found in {data_path}', 'red'))
    
    # Get the children of the first test-fold directory
    subject_paths = get_all_from_directory(test_subject_paths[0], return_full_path = True)
    if not subject_paths:
        raise Exception(colored(f'Error: no files found in {test_subject_paths[0]}', 'red'))
    
    # The inner loop will have 'config' in its directories
    config_count = [p for p in subject_paths if 'config' in p.split('/')[-1]]
    config_count = [p for p in config_count if os.path.isdir(p)]
    if len(config_count) > 1:
        print(colored("Data detected as inner loop.", 'purple'))
        return False
    print(colored("Data detected as outer loop.", 'purple'))
    
    return True



def get_subfolds(data_path):
    """
    This function will get every "subfold" that exists.

    Args:
        data_path (str): The path to a data directory.

    Returns:
        dict: Of paths organized by structure.
    """
    
    # Stores paths in a dictionary
    subfolds = {}

    # Gets the test subjects
    test_subject_paths = get_all_from_directory(data_path, return_full_path = True)

    # For each subject, gets the configurations
    for subject in test_subject_paths:
        config_paths = get_all_from_directory(subject, return_full_path = True)
        subject_id = subject.split('/')[-1].split('_')[-1]

        # For each model/config, find its contents
        for config in config_paths:

            # Checks if the model has contents
            subfold_paths = get_all_from_directory(config, return_full_path = True)
            
            # Add check for if this contains info for 1 fold.
            info = {'file_name': False, 'model': False, 'prediction': False, 'true_label': False}
            for path in subfold_paths:
                subpath = path.split('/')[-1]
                if subpath in info:
                    info[subpath] = True
                    
            if subfold_paths:

                # Check that the dictionary contains the model/subject
                model_name = subfold_paths[0].split('/')[-1].split('_')[0]
                if model_name not in subfolds:
                    subfolds[model_name] = {}
                if subject_id not in subfolds[model_name]:
                    subfolds[model_name][subject_id] = []

                # Add to results
                if False in list(info.values()):
                    subfolds[model_name][subject_id].extend(subfold_paths)
                else:
                    subfolds[model_name][subject_id].append(config)

    # Return the directory-paths
    return subfolds



def process_subfold(subfold, target_folder, is_csv, is_validation, is_testing, is_index, is_outer):
    """
    Process a single subfold and return relevant files.
    
    TODO
    """
    
    # Checks if target folder exists
    subfiles = get_all_from_directory(subfold, return_full_path = False)
    
    if target_folder not in subfiles:
        print(colored(f"Warning: {target_folder} not detected in {subfold}.", "yellow"))
        return []


    # Gets the files from the target folder
    full_target_path = os.path.join(subfold, target_folder)
    target_paths = get_all_from_directory(full_target_path, return_full_path = True)
    
    # Apply filters
    if is_csv:
        target_paths = [f for f in target_paths if f.endswith(".csv")]
    
    target_paths = filter_files_by_type(target_paths, is_validation, is_testing, is_outer)
    
    if target_folder == "prediction":
        target_paths = filter_files_for_predictions(target_paths)
    
    #target_paths = filter_files_by_index(target_paths, is_index)

    return target_paths



def get_subfolder_files(data_path, target_folder, is_index=None, get_validation=False, get_testing=False, is_csv=True, return_is_outer=False, is_outer=None):
    """
    Get a set of files from each "subfold" within a particular target directory.

    Args:
        data_path (str): Path to the data directory.
        target_folder (str): Name of the target folder to search in.
        is_index (bool, optional): Whether to return indexed or probability results.
        get_validation (bool, optional): Get validation files only.
        get_testing (bool, optional): Get test files only.
        is_csv (bool, optional): Check if the file should be a CSV file.
        return_is_outer (bool, optional): Return whether the data is in outer loop format.
        is_outer (bool, optional): Specification if this data path is the outer loop.

    Returns:
        dict: Paths organized by structure.
        bool: True if the directory is an outer loop (if return_is_outer is True).
    """
    
    # Initializations
    target_subfolder_files = {}
    is_outer = is_outer if is_outer is not None else is_outer_loop(data_path)

    # Set both 'gets' to true if both are false
    if not get_validation and not get_testing:
        get_validation = get_testing = True

    
    # Gets subfolds
    subfolds = get_subfolds(data_path)
    
    
    for model_name, subjects in subfolds.items():
        
        # Adds the model dictionary to the global dictionary
        target_subfolder_files[model_name] = {}
        
        for subject_id, folds in subjects.items():
            
            # Adds the current subject dictionary to the model dictionary
            target_subfolder_files[model_name][subject_id] = []

            # Checks for correct directory structure
            if target_folder in [f.split('/')[-1] for f in subfolds[model_name]] or \
               target_folder in [f.split('/')[-1] for f in folds]:
                raise Exception(colored("Error: Incorrect data format. Levels should be model->subject->fold.", 'red'))

            for subfold in folds:
                
                files = process_subfold(subfold, target_folder, is_csv, get_validation, get_testing, is_index, is_outer)
                
                target_subfolder_files[model_name][subject_id].extend(files)
    
    """
    # Handle case when no indexed predictions are found
    if target_folder == 'prediction' and is_index:
        test_config_key = next(iter(target_subfolder_files))
        test_fold_key = next(iter(target_subfolder_files[test_config_key]))
        if not target_subfolder_files[test_config_key][test_fold_key]:
            print(colored('No indexed-predictions found. Running formatter...', 'yellow'))
            reformat(data_path, is_outer=is_outer)
            target_subfolder_files = get_subfolder_files(data_path, target_folder, is_index, get_validation, get_testing, is_csv, is_outer=is_outer)
            if not target_subfolder_files[test_config_key][test_fold_key]:
                raise Exception(colored("Error: No prediction files found. Check inputs and run predicted_formatter.py", 'red'))
    """
    return (target_subfolder_files, is_outer) if return_is_outer else target_subfolder_files



def get_history_paths(data_path):
    """
    This function will get every history file from each model-fold.

    Args:
        data_path (str): Path to the data directory.

    Returns:
        dict: Of paths organized by structure.
    """
    
    # Will return a dictionary of arrays separated by model
    histories = {}

    # Get the existing subfolds
    subfolds = get_subfolds(data_path)

    # Iterate through the subfolds
    for model_name in subfolds:
        histories[model_name] = {}
        for subject_id in subfolds[model_name]:
            histories[model_name][subject_id] = []

            # Each subfold should contain one history file. Try to find it.
            for subfold in subfolds[model_name][subject_id]:
                subfiles = get_all_from_directory(subfold, return_full_path = True)

                # Search for the history file
                missing = True
                for subfile in subfiles:
                    if subfile.endswith("history.csv"):
                        histories[model_name][subject_id].append(subfile)
                        missing = False
                        break

                # Warn that the target file was not detected
                if missing:
                    print(colored("Warning: a history file was not detected in " + subfold, "yellow"))
                    continue

    # Return results
    return histories



def get_config_indexes(data_path):
    """
    Returns the config names and their index (E.x. config 1 -> resnet) 

    Args:
        data_path (str): Path to the data directory.

    Returns:
        dict: A model-index dictionary.
    """
    
    # Returns a dict
    results = {}

    # The test folds
    test_subject_paths = get_all_from_directory(data_path, return_full_path = True)

    # For each subject, get the configurations
    for subject in test_subject_paths:
        config_paths = get_all_from_directory(subject, return_full_path = True)

        # For each model/config, find its contents
        for config in config_paths:
            config_id = config.split("/")[-1].split("_")[1]

            # Check if the model has contents
            subfold_paths = get_all_from_directory(config, return_full_path = True)
            if subfold_paths:
                # Check that the dictionary contains the model/subject
                model_name = subfold_paths[0].split('/')[-1].split('_')[0]
                results[model_name] = config_id
                
    return results