import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from src.data_processing.create_data_csv import create_data_csv
from src.setup.command_line_parser import command_line_parser
from src.setup.get_training_configs_list import get_training_configs_list

if __name__ == "__main__":

    # Parses the command line arguments
    command_line_arguments = command_line_parser()
    
    # Gets the configuration file
    list_of_configs, _ = get_training_configs_list(command_line_arguments['file'], command_line_arguments['folder'])
    configuration_file = list_of_configs[0]
    
    
    # Creates the path where to put the CSV files
    data_path = os.path.join(PROJECT_ROOT, 'data')
    
    # Creates the header
    csv_header = ["files", "labels"]
    
    # Creates the CSV files
    create_data_csv(data_path, configuration_file, csv_header)
