import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from src.file_processing.file_getter import get_subfolder_files


if __name__ == "__main__":
    """
    Executes Program.
    """
    
    data_path = "results/distributed/pig_kidney_subset_parallel/training_results"
    
    subfolder_files = get_subfolder_files(data_path, "true_label", is_index = True, get_validation = True, get_testing = False, is_outer = False)

    print(subfolder_files)
    