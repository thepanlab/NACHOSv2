import os
import sys
from termcolor import colored
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from src.file_processing.file_getter import get_subfolder_files
from src.results_processing.confusion_matrix import confusion_matrix
from src.results_processing.results_processing_utils.get_config import parse_json


def find_directories(data_path, is_outer):
    """
    Finds the directories for every input needed to make graphs.

    Args:
        data_path (string): The path of the data directory.
        is_outer (bool): Whether the path is of the outer loop.

    Returns:
        dict: Two dictionaries of prediction and truth paths.
    """
    
    # Get the paths of every prediction and true CSV, as well as the fold-names
    predictions_paths = get_subfolder_files(data_path, "prediction", is_index=True, get_validation=True, get_testing=False, is_outer=is_outer)
    true_labels_paths = get_subfolder_files(data_path, "true_label", is_index=True, get_validation=True, get_testing=False, is_outer=is_outer)
    
    return predictions_paths, true_labels_paths



def run_program(args, predictions_paths, true_labels_paths):
    """
    Runs the program for each item.

    Args:
        args (dict): A JSON configuration input as a dictionary.
        predictions_paths (dict): A dictionary of prediction paths for the given data directory.
        true_labels_paths (dict): A dictionary of truth paths for the given data directory.
    """
    
    # Gets the proper file names for output
    json = {
        label: args[label] for label in (
            'label_types', 'output_path'
        )
    }

    # For each item, runs the program
    for model in predictions_paths:
        
        for subject in predictions_paths[model]:
            
            for item in range(len(predictions_paths[model][subject])):
                
                try:
                    # Get the program's arguments
                    json = generate_json(predictions_paths, true_labels_paths, model, subject, item, json)
                    
                    if json is not None:
                        confusion_matrix.main(json)

                # Catch weird stuff
                except Exception as err:
                    print(colored(f"Exception caught. Double-check your configuration is of the outer loop or not.\n\t{str(err)}", "red"))
                    print(colored(f"{traceback.format_exc()}\n", "yellow"))



def generate_json(predictions_paths, true_labels_paths, config, subject, item, json):
    """
    Creates a dictionary of would-be JSON arguments.

    Args:
        predictions_paths (dict): A dictionary of prediction paths.
        true_labels_paths (dict): A dictionary of truth paths.
        config (str): The config (model) of the input data.
        subject (str): The subject (test fold) of the input data.
        item (int): The item's index in the paths dictionary-array.
        json (dict): A dictionary with some values already added.

    Returns:
        dict: A JSON configuration as a dict.
    """
    
    # The current expected suffix format for true labels
    true_label_suffix = " true label index.csv"

    # Creates dictionary for every item
    json["pred_path"] = predictions_paths[config][subject][item]
    json["true_path"] = true_labels_paths[config][subject][item]
    json["output_file_prefix"] = true_labels_paths[config][subject][item].split('/')[-1].replace(true_label_suffix, "")
    
    return json



def main():
    """
    The Main Program.
    """
    
    # Gets program configuration and run using its contents
    config = parse_json(os.path.abspath('src/results_processing/confusion_matrix/confusion_matrix_many_config.json'))
    
    predictions_paths, true_labels_paths = find_directories(config["data_path"], config['is_outer'])
    
    run_program(config, predictions_paths, true_labels_paths)



if __name__ == "__main__":
    """
    Executes Program.
    """
    
    main()
