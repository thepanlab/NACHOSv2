import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from src.results_processing.learning_curve.create_graphs import create_graphs
from src.results_processing.learning_curve.learning_curve_utils import file_verification
from src.results_processing.results_processing_utils.get_configuration_file import parse_json


def main(config = None):
    """
    Plots the learning curves for the given file.

    Args:
        config (dict, optional): A custom configuration. Defaults to None.
    """
    
    # Obtains the dictionary of configurations from a json file
    if config is None:
        config = parse_json(os.path.join(os.path.dirname(__file__), 'learning_curve_config.json'))

    
    # Grabs the file directories 
    file_path = config['input_path']
    results_path = config['output_path']
    
    # Makes sure the result path exists
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    
    # Creates the list of files
    list_files = os.listdir(file_path)
    verified_list_files = file_verification(list_files)
    
    
    # Creates the graphs
    create_graphs(verified_list_files, file_path, results_path, config)



if __name__ == "__main__":
    """
    Plots the learning curves for a single file.
    """
    
    main()
    