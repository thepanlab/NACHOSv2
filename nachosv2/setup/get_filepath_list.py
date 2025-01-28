from typing import List
from pathlib import Path

def get_filepath_list(directory_to_search_path: Path,
                 string_in_filename: str) -> List[Path]:
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
    
    # Using rglob to recursively search for files
    for file in directory_to_search_path.rglob(f'*{string_in_filename}*.csv'):
        matched_files.append(file)
    
    return matched_files