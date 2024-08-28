import math
import os
import pandas as pd
import sys
from termcolor import colored

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from src.results_processing.confusion_matrix.get_input_matrices import get_input_matrices
from src.results_processing.confusion_matrix.confusion_matrix_utils import get_matrices_of_mode_shape
from src.results_processing.results_processing_utils.get_configuration_file import parse_json


def prepare_matrix_dataframes(labels):
    """
    Prepares empty DataFrames for various matrix calculations.
    
    Args:
        labels (list): List of labels for matrix rows and columns.
    
    Returns:
        tuple: Tuple of DataFrames (matrix_avg, matrix_perc_avg, matrix_sum_temp)
    """
    
    index_labels = [["Truth"] * len(labels), labels]
    columns_labels = [["Predicted"] * len(labels), labels]
    
    matrix_avg = pd.DataFrame(0.0, columns=labels, index=labels)
    matrix_perc_avg = pd.DataFrame(0.0, columns=labels, index=labels)
    matrix_sum_temp = pd.DataFrame(0.0, columns=labels, index=labels)
    
    for matrix in [matrix_avg, matrix_perc_avg, matrix_sum_temp]:
        matrix.index = index_labels
        matrix.columns = columns_labels
    
    return matrix_avg, matrix_perc_avg, matrix_sum_temp



def calculate_matrix_averages(valid_matrices, all_matrices, labels):
    """
    Calculates the average and percentage average matrices.
    
    Args:
        valid_matrices (list): List of valid matrix file paths.
        all_matrices (list): List of all matrix file paths.
        labels (list): List of labels for matrix rows and columns.
    
    Returns:
        tuple: Tuple of DataFrames (matrix_avg, matrix_perc_avg, perc_values)
    """
    matrix_avg, matrix_perc_avg, matrix_sum_temp = prepare_matrix_dataframes(labels)
    perc_values = {}
    
    for val_index, val_matrix in enumerate(valid_matrices):
        val_fold = pd.read_csv(val_matrix, index_col=[0,1], header=[0,1])
        for row in labels:
            row_total = sum(val_fold["Predicted"][col]["Truth"][row] for col in labels)
            for col in labels:
                item = val_fold["Predicted"][col]["Truth"][row]
                matrix_avg["Predicted"][col]["Truth"][row] += item
    
    for val_index, val_matrix in enumerate(all_matrices):
        val_fold = pd.read_csv(val_matrix, index_col=[0,1], header=[0,1])
        perc_values[val_index] = pd.DataFrame(0, columns=labels, index=labels)
        perc_values[val_index].index = matrix_avg.index
        perc_values[val_index].columns = matrix_avg.columns
        
        for row in labels:
            row_total = sum(val_fold["Predicted"][col]["Truth"][row] for col in labels)
            for col in labels:
                item = val_fold["Predicted"][col]["Truth"][row]
                matrix_sum_temp["Predicted"][col]["Truth"][row] += item
                weighted_item = item / row_total if row_total != 0 else 0
                perc_values[val_index]["Predicted"][col]["Truth"][row] = weighted_item
                matrix_perc_avg["Predicted"][col]["Truth"][row] += weighted_item
    
    n_items_valid = len(valid_matrices)
    n_items_all = len(all_matrices)
    matrix_avg /= n_items_valid
    matrix_perc_avg /= n_items_all
    
    return matrix_avg, matrix_perc_avg, perc_values



def calculate_matrix_errors(valid_matrices, all_matrices, matrix_avg, matrix_perc_avg, perc_values, labels):
    """
    Calculates the standard error matrices.
    
    Args:
        valid_matrices (list): List of valid matrix file paths.
        all_matrices (list): List of all matrix file paths.
        matrix_avg (DataFrame): Average matrix.
        matrix_perc_avg (DataFrame): Percentage average matrix.
        perc_values (dict): Dictionary of percentage values.
        labels (list): List of labels for matrix rows and columns.
    
    Returns:
        tuple: Tuple of DataFrames (matrix_err, matrix_perc_err)
    """
    matrix_err, matrix_perc_err, _ = prepare_matrix_dataframes(labels)
    
    for val_matrix in valid_matrices:
        val_fold = pd.read_csv(val_matrix, index_col=[0,1], header=[0,1])
        for row in labels:
            for col in labels:
                true = val_fold["Predicted"][col]["Truth"][row]
                mean = matrix_avg["Predicted"][col]["Truth"][row]
                matrix_err["Predicted"][col]["Truth"][row] += (true - mean) ** 2
    
    for val_index, val_matrix in enumerate(all_matrices):
        for row in labels:
            for col in labels:
                true_weighted = perc_values[val_index]["Predicted"][col]["Truth"][row]
                mean_weighted = matrix_perc_avg["Predicted"][col]["Truth"][row]
                matrix_perc_err["Predicted"][col]["Truth"][row] += (true_weighted - mean_weighted) ** 2
    
    n_items_valid = len(valid_matrices)
    n_items_all = len(all_matrices)
    
    for row in labels:
        for col in labels:
            if n_items_valid > 1:
                matrix_err["Predicted"][col]["Truth"][row] = math.sqrt(matrix_err["Predicted"][col]["Truth"][row] / (n_items_valid - 1)) / math.sqrt(n_items_valid)
            if n_items_all > 1:
                matrix_perc_err["Predicted"][col]["Truth"][row] = math.sqrt(matrix_perc_err["Predicted"][col]["Truth"][row] / (n_items_all - 1)) / math.sqrt(n_items_all)
    
    return matrix_err, matrix_perc_err



def create_combo_matrices(matrix_avg, matrix_err, matrix_perc_avg, matrix_perc_err, labels, round_to):
    """
    Creates combination matrices of mean and standard error.
    
    Args:
        matrix_avg (DataFrame): Average matrix.
        matrix_err (DataFrame): Error matrix.
        matrix_perc_avg (DataFrame): Percentage average matrix.
        matrix_perc_err (DataFrame): Percentage error matrix.
        labels (list): List of labels for matrix rows and columns.
        round_to (int): Number of decimal places to round to.
    
    Returns:
        tuple: Tuple of DataFrames (matrix_combo, matrix_perc_combo)
    """
    matrix_combo = pd.DataFrame('', columns=labels, index=labels)
    matrix_perc_combo = pd.DataFrame(0, columns=labels, index=labels)
    matrix_combo.index = matrix_perc_combo.index = matrix_avg.index
    matrix_combo.columns = matrix_perc_combo.columns = matrix_avg.columns
    
    for row in labels:
        for col in labels:
            matrix_combo.loc[("Truth", row), ("Predicted", col)] = f'{round(matrix_avg["Predicted"][col]["Truth"][row], round_to)} ± {round(matrix_err["Predicted"][col]["Truth"][row], round_to)}'
            if not matrix_perc_avg.empty:
                matrix_perc_combo.loc[("Truth", row), ("Predicted", col)] = f'{round(matrix_perc_avg["Predicted"][col]["Truth"][row], round_to)} ± {round(matrix_perc_err["Predicted"][col]["Truth"][row], round_to)}'
    
    return matrix_combo, matrix_perc_combo



def save_matrices(output_folder, configuration, test_fold, matrix_avg, matrix_err, matrix_combo, matrix_perc_avg, matrix_perc_err, matrix_perc_combo, round_to):
    """
    Saves all calculated matrices to CSV files.
    
    Args:
        output_folder (str): Folder to save the matrices.
        configuration (str): Configuration name.
        test_fold (str): Test fold name.
        matrix_avg (DataFrame): Average matrix.
        matrix_err (DataFrame): Error matrix.
        matrix_combo (DataFrame): Combination matrix.
        matrix_perc_avg (DataFrame): Percentage average matrix.
        matrix_perc_err (DataFrame): Percentage error matrix.
        matrix_perc_combo (DataFrame): Percentage combination matrix.
        round_to (int): Number of decimal places to round to.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    matrix_avg.round(round_to).to_csv(os.path.join(output_folder, f'{configuration}_{test_fold}_conf_matrix_mean.csv'))
    matrix_err.round(round_to).to_csv(os.path.join(output_folder, f'{configuration}_{test_fold}_conf_matrix_stderr.csv'))
    matrix_combo.to_csv(os.path.join(output_folder, f'{configuration}_{test_fold}_conf_matrix_mean_stderr.csv'))
    
    if not matrix_perc_avg.empty:
        matrix_perc_avg.round(round_to).to_csv(os.path.join(output_folder, f'{configuration}_{test_fold}_conf_matrix_perc_mean.csv'))
        matrix_perc_err.round(round_to).to_csv(os.path.join(output_folder, f'{configuration}_{test_fold}_conf_matrix_perc_stderr.csv'))
        matrix_perc_combo.to_csv(os.path.join(output_folder, f'{configuration}_{test_fold}_conf_matrix_perc_mean_stderr.csv'))
    
    print(colored(f"Mean confusion matrix results created for {test_fold} in {configuration}", 'green'))



def get_mean_matrices(matrices, shapes, output_path, labels, round_to, is_outer):
    """
    This function gets the mean confusion matrix of every inner loop.

    Args:
        matrices (dict): A dictionary of matrices organized by configuration and testing fold.
        shapes (dict): A dictionary of shapes organized by configuration and testing fold.
        output_path(string): The path to write the average matrices to.
        labels(list(str)): The labels of the matrix data.
        round_to (int): Number of decimal places to round to.
        is_outer (bool): Whether this data is of the outer loop.
    """
    
    if isinstance(labels, dict):
        labels = list(labels.keys())
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    all_valid_matrices = get_matrices_of_mode_shape(shapes, matrices, is_outer)

    for configuration in matrices:
        for test_fold in matrices[configuration]:
            valid_matrices = [all_valid_matrices[configuration][t][i] for t in all_valid_matrices[configuration] for i in range(len(all_valid_matrices[configuration][t]))] if is_outer else all_valid_matrices[configuration][test_fold]
            all_matrices = matrices[configuration][test_fold]

            if len(valid_matrices) <= 1:
                print(colored(f"Warning: Mean calculation skipped for testing fold {test_fold} in {configuration}. Must have multiple validation folds.\n", 'yellow'))
                continue

            matrix_avg, matrix_perc_avg, perc_values = calculate_matrix_averages(valid_matrices, all_matrices, labels)
            matrix_err, matrix_perc_err = calculate_matrix_errors(valid_matrices, all_matrices, matrix_avg, matrix_perc_avg, perc_values, labels)
            matrix_combo, matrix_perc_combo = create_combo_matrices(matrix_avg, matrix_err, matrix_perc_avg, matrix_perc_err, labels, round_to)

            output_folder = os.path.join(output_path, f'{configuration}_{test_fold}/')
            save_matrices(output_folder, configuration, test_fold, matrix_avg, matrix_err, matrix_combo, matrix_perc_avg, matrix_perc_err, matrix_perc_combo, round_to)
    


def main():
    """
    The Main Program.
    """
    
    # Gets the configuration file
    configuration = parse_json(os.path.abspath('src/results_processing/confusion_matrix/confusion_matrix_many_means_config.json'))
    
    # Gets the input informations
    matrices, shapes = get_input_matrices(configuration['matrices_path'], configuration['is_outer'])
    
    get_mean_matrices(matrices, shapes, configuration['means_output_path'], configuration['label_types'], configuration['round_to'], configuration['is_outer'])



if __name__ == "__main__":
    main()