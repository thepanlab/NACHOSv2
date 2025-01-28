import os
import sys
from pathlib import Path

from termcolor import colored
import pandas as pd
from sklearn import metrics
import re

from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.get_config_list import get_config_list
from nachosv2.setup.get_filepath_list import get_filepath_list
from nachosv2.setup.get_class_names_from_prediction_path import get_class_names_from_prediction_path

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

# TODO
def generate_individual_confusion_matrix(config: dict,
                                         path: Path) -> pd.DataFrame:
    
    df_results = pd.read_csv(path)
    actual = df_results['true_label']
    predicted = df_results['predicted_class']
    
    confusion_matrix = metrics.confusion_matrix(actual, predicted) 

    df_class_names = get_class_names_from_prediction_path(config, path)
    df_class_names.set_index('index', inplace=True)
    series_class_names = df_class_names['class_name']
    series_class_names.name = None
    
    # Transform to dataframe with column names
    row_indices = pd.MultiIndex.from_product([['Ground truth'], series_class_names],
                                                 names=[None, None])
    column_indices = pd.MultiIndex.from_product([['Predicted'], series_class_names],
                                                   names=[None, None])
    
    confusion_matrix_df = pd.DataFrame(data=confusion_matrix,
                                       index=row_indices,
                                       columns=column_indices)
    
    
    return confusion_matrix_df


def get_filepath_confusion_matrix(config: dict,
                                  path: Path):
    """
    Get the name of the confusion matrix file.

    Args:
        path (Path): The path of the confusion matrix file.

    Returns:
        string: The name of the confusion matrix file.
    """
    
    file_name = path.name.replace("prediction_results", "confusion_matrix")
    folder_path = Path(config["output_path"])
    folder_path.mkdir(mode=0o777, parents=True, exist_ok=True)
    
    filepath = folder_path / file_name
    
    return filepath


def main():
    """
    The Main Program.
    """
    args = parse_command_line_args()
    # Gets program configuration and run using its contents
    # config_dict_list is a list of dictionaries
    config_dict_list = get_config_list(args['file'], args['folder'])
    
    # TODO
    # Look for csv files with specific format
    for config in config_dict_list:
        if not Path(config['data_path']).exists():
            raise Error(print(colored(f"Path {config['data_path']} does not exist.", "red")))
        folder_training_results_path = Path(config['data_path'])
        
        string_filename = "prediction_results"
        predictions_file_path_list = get_filepath_list(folder_training_results_path,
                                                       string_filename)
        
        # Regex to match test and validation info
        pattern = r"test_([A-Za-z0-9]+)_val_([A-Za-z0-9]+)"

        # Extract and print test and validation fold numbers
        for path in predictions_file_path_list:
            filename = path.name
            match = re.search(pattern, filename)
            if match:
                test_fold, val_fold = match.groups()
                print(f"File: {filename}")
                print(f"  Test fold: {test_fold}")
                print(f"  Validation fold: {val_fold}")
                print("-----")
                
        # TODO: get class name files by modifying filename
        # 
            cf_df = generate_individual_confusion_matrix(config, path)

            cf_filepath = get_filepath_confusion_matrix(config,
                                                        path)
            cf_df.to_csv(cf_filepath)
        
        # Add column to specify the predicted and the ground truth    
        # TODO: make confusion matrix for all the validation
        # verify that all have the same amount of files
        
     
    predictions_paths, true_labels_paths = find_directories(config["data_path"],
                                                            config['is_outer'])
    
    run_program(config, predictions_paths, true_labels_paths)


if __name__ == "__main__":
    main()
