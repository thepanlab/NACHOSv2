
import os
import regex as re

from src.file_processing.get_from_directory import get_files_from_directory


def extract_fold_info(filename, is_outer):
    """
    Extracts test and validation fold information from filename.
    
    Args:
        filename (str): Name of the file.
        is_outer (bool): Whether this is outer loop data.
    
    Returns:
        tuple: test_fold, validation_fold (if inner), shape
    """
    
    # Inner loop
    if not is_outer:
        shape = re.findall(r'\d+', filename)[-1]
        test_fold = re.search('_test_.*_val_', filename).captures()[0].split("_")[2]
        validation_fold = re.search('_test_.*_val_', filename).captures()[0].split("_")[4]
        
    
    # Outer loop
    else:
        shape = int(re.search('_.*_conf_matrix.csv', filename).captures()[0].split("_")[1])
        test_fold = re.search('_test_.*_', filename).captures()[0].split("_")[2]
        validation_fold = None
    
    
    return test_fold, validation_fold, shape



def organize_paths(all_paths, matrices_path, is_outer):
    """
    Organizes paths by configuration and test/validation folds.
    
    Args:
        all_paths (list): List of all file paths.
        matrices_path (str): Base path for matrices.
        is_outer (bool): Whether this is outer loop data.
    
    Returns:
        tuple: Organized paths and shapes dictionaries.
    """
    
    # Initializations
    organized_paths = {}
    organized_shapes = {}

    
    for current_path in all_paths:
        
        # Initializations
        configuration = current_path.split("/")[0].split('_')[0]
        
        if configuration not in organized_paths:
            organized_paths[configuration] = {}
            organized_shapes[configuration] = {}
        
        filename = current_path.split('/')[-1]
        
        
        test_fold, validation_fold, shape = extract_fold_info(filename, is_outer)
        
        
        # Inner loop
        if not is_outer: 
            
            # Creates the dictionaries
            if test_fold not in organized_paths[configuration]:
                organized_paths[configuration][test_fold] = {}
                organized_shapes[configuration][test_fold] = {}
                
            organized_shapes[configuration][test_fold][validation_fold] = shape
            organized_paths[configuration][test_fold][validation_fold] = os.path.join(matrices_path, current_path)
        
        # Outer loop
        else:
            organized_shapes[configuration][test_fold] = shape
            organized_paths[configuration][test_fold] = os.path.join(matrices_path, current_path)


    return organized_paths, organized_shapes



def get_input_matrices(matrices_path, is_outer):
    """
    Finds the existing configs, test folds, and validations folds of all matrices.

    Args:
        matrices_path (string): The path of the matrices' directory.
        is_outer (bool): Whether this data is of the outer loop.

    Returns:
        dict: A dictionary of matrix-dataframes and their prediction shapes, organized by configuration and testing fold.
    """

    # Gets the matrix files
    all_paths = get_files_from_directory(matrices_path)

    # Organizes the paths
    input_matrices = organize_paths(all_paths, matrices_path, is_outer)

    
    return input_matrices
