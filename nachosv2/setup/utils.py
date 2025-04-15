# utils.py

from pathlib import Path
from typing import Optional, List
from termcolor import colored
import pandas as pd
import re


def get_other_result(path: Path,
                     suffix: str):
    # take path prediction and get class name file
    # other_path = Path(str(path).replace('prediction_results',
    #                                      suffix))
    
    other_path = Path(re.sub(r'prediction_(?:test|validation)', 'class_names', str(path)))
    # verify other_path exists
    if not other_path.exists():
        raise FileNotFoundError(f"The file {other_path} does not exist.")
    
    df_other = pd.read_csv(other_path,
                           index_col=0)
    
    return df_other


def get_filepath_list(directory_to_search_path: Path,
                      string_in_filename: str,
                      is_cv_loop: bool) -> List[Path]:
    """
    Recursively searches for CSV files containing 'string_in_filename'
    in their filenames within a given directory.

    Args:
    directory (Path): The path to the directory where the search will start.

    Returns:
    List[Path]: A list of Paths to the matching CSV files.
    """
    # Check if the given directory is indeed a directory
    if not directory_to_search_path.is_dir():
        raise ValueError("The provided path is not a directory.")
    
    # List to store the paths of matching CSV files
    matched_files = []
    
    loop_folder = "CV" if is_cv_loop else "CT"
    search_pattern = f'**/{loop_folder}/**/*{string_in_filename}*.csv'

    # Using rglob to recursively search for files
    for file in directory_to_search_path.rglob(search_pattern):
        matched_files.append(file)
    
    return matched_files


def determine_if_cv_loop(path: Path):
    """Based on the filepath name of results we can determine if it is a cross-validation loop or cross-testing loop

    Args:
        path (Path): any results filepath

    Returns:
        bool: if it is cv loop
    """
    if "_val_" in path.name:
        return True
    else:
        return False
    
    
def get_default_folder(path: Path,
                       folder_name: str,
                       is_cv_loop: bool) -> Path:
    """
    Constructs a default output folder path based on the location of cross-validation (CV) 
    or cross-testing (CT) results.

    The function searches for either a "CT" or "CV" directory in the given path. If the path 
    is a CSV file, it looks for "CT" or "CV" in the path parts and builds the output folder 
    path accordingly. If the path is a directory, it recursively searches for a "CT" or "CV" 
    subfolder. The resulting output path will be `{CT or CV}/{folder_name}`.

    Args:
        path (Path): A filepath  pointing to a result file (e.g., history CSV) 
                     or folder structure containing training outputs.
        folder_name (str): Name of the folder to create or point to under the detected "CT" or "CV" directory.
        is_cv_loop (bool): Indicates whether to look for the "CV" folder (if True) or "CT" folder (if False).

    Returns:
        Path: The constructed path to the output folder under "CT" or "CV".

    Raises:
        FileNotFoundError: If the function fails to locate a "CT" or "CV" directory in the given path.
    """

    default_folder_path = None
    subpath_to_find = "CV" if is_cv_loop else "CT"
    
    if path.suffix == ".csv":
        # Search for the index of 'CT' or 'CV' in the file path components
        # e.g. /home/pcallec/NACHOS/results/pig_kidney_subset/CT/training_results/test_k1/hpconfig_0/test_k1_hpconfig_0_prediction_results.csv
        index_training_results = None
        for i, part in enumerate(path.parts):
            if part in ("CT", "CV"):
                index_training_results = i
                break
            
        # Construct base path up to 'CT' or 'CV'
        # e.g. /home/pcallec/NACHOS/results/pig_kidney_subset/CT
        base_path = Path(*path.parts[:index_training_results + 1])
        # e.g. /home/pcallec/NACHOS/results/pig_kidney_subset/CT/new_folder
        default_folder_path = base_path / folder_name
    else:
        # Search recursively in the directory for a subfolder named 'CT' or 'CV'
        for subpath in path.rglob("*"):  # rglob searches recursively for given pattern
            if subpath.is_dir() and (subpath.name == subpath_to_find):
                # Path found, get the parent of 'training_results'
                default_folder_path = subpath / folder_name
                break
    
    if default_folder_path is None:
        raise FileNotFoundError(f"Could not find 'training_results' folder in the path {path}.")
    
    return default_folder_path


def get_folder_path(output_path: Path,
                    folder_name: str,
                    is_cv_loop: bool):
    loop = "CV" if is_cv_loop else "CT"
    folder_path = output_path / loop / folder_name
    folder_path.mkdir(mode=0o775, parents=True, exist_ok=True)

    return folder_path


def get_newfilepath_from_predictions(predictions_filepath: Path,
                                     suffix_name: str,
                                     is_cv_loop: bool,
                                     output_path: Optional[Path]):
    """
    Get the name of the confusion matrix file.

    Args:
        path (Path): The path of the confusion matrix file.

    Returns:
        string: The name of the confusion matrix file.
    """
    
    file_name = predictions_filepath.name.replace("prediction",
                                                  suffix_name)
    if output_path is None:
        folder_path = get_default_folder(predictions_filepath,
                                         suffix_name,
                                         is_cv_loop)
        print(colored(f"\nUsing default folder: {folder_path}.", 'magenta'))
    else:
        folder_path = output_path
        
    folder_path.mkdir(mode=0o777, parents=True, exist_ok=True)
    
    filepath = folder_path / file_name
    
    return filepath


def get_filepath_from_results_path(results_path: Path,
                                   folder_name: str,
                                   file_name: str,
                                   is_cv_loop: bool,
                                   output_path: Optional[Path] = None) -> Path:

    if output_path is None:
        folder_path = get_default_folder(results_path,
                                         folder_name,
                                         is_cv_loop)
        
        print(colored(f"\nUsing default folder: {folder_path}.", 'magenta'))
    else:
        folder_path = output_path
    
    folder_path.mkdir(mode=0o777, parents=True, exist_ok=True)

    filepath = folder_path / file_name
    
    return filepath