import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
from termcolor import colored

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from src.results_processing.confusion_matrix.confusion_matrix_utils import check_and_convert_labels, check_and_convert_values
from src.results_processing.results_processing_utils.get_config import parse_json
from src.results_processing.results_processing_utils.get_data import get_data


def create_confusion_matrix(true_vals, pred_vals, results_path, file_name, labels):
    """
    Creates confusion matrix and saves as a csv in the results directory.

    Args:
        true_vals (pandas.Dataframe): An array of true values.
        pred_vals (pandas.Dataframe): An array of predicted values.
        results_path (str): The path to write a matrix to.
        file_name (str): The name of the matrix file.
        labels (list(str)): A list of the labels associated with the classification index values.

    Raises:
        Exception: When true and predicted values are not equal in length.

    Returns:
        pandas.Dataframe: The created confusion matrix.
    """
    
    # Checks if the input is valid
    if len(true_vals) != len(pred_vals):
        raise Exception(colored(f'The length of true and predicted values are not equal: \n' +
                                f'\tTrue: {len(true_vals)} | Predicted: {len(pred_vals)}', 'red'))
    
    
    # Checks the values and converts if necessary
    labels_list = check_and_convert_labels(labels)

    true_vals = check_and_convert_values(true_vals, labels_list)
    
    pred_vals = check_and_convert_values(pred_vals, labels_list)
    
    
    # Creates the matrix
    conf_matrix = confusion_matrix(true_vals, pred_vals, labels = labels_list)
    
    
    # Saves it to a panda dataframe
    conf_matrix_df = pd.DataFrame(conf_matrix, columns = labels_list, index = labels_list)

    # Creates the extra-index/col names
    conf_matrix_df.index = [["Truth"] * len(labels_list), conf_matrix_df.index]
    conf_matrix_df.columns = [["Predicted"] * len(labels_list), conf_matrix_df.columns]


    # Outputs the results to a CSV
    conf_matrix_df.to_csv(f"{os.path.splitext(os.path.join(results_path, file_name))[0]}_{len(pred_vals)}_conf_matrix.csv")


    print(colored("Confusion matrix created for " + file_name, 'green'))
    
    
    return conf_matrix



def main(config = None):
    """
    The main program.

    Args:
        config (dict): A JSON configuration as a dictionary. (Optional)
    """
    
    # Checks that the output path exists
    if config is None:
        config = parse_json(os.path.join(os.path.dirname(__file__), 'confusion_matrix_config.json'))
    
    
    # Initializations
    output_path = config['output_path']
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    # Obtains needed labels and predictions
    predictions_file_path = config['pred_path']
    true_labels_file_path = config['true_path']
    
    if not os.path.exists(predictions_file_path):
        raise Exception(colored("Error: The prediction path is not valid: " + predictions_file_path, 'red'))
    
    if not os.path.exists(true_labels_file_path):
        raise Exception(colored("Error: The true-value path is not valid: " + true_labels_file_path, 'red'))
    
    true_val, pred_val = get_data(predictions_file_path, true_labels_file_path)
    
    
    # Creates the confusion matrix
    create_confusion_matrix(true_val, pred_val, output_path, config['output_file_prefix'], config['label_types'])



if __name__ == "__main__":
    """
    Executes the program.
    """
    
    main()
