import math
import os
import pandas as pd
import regex as re
from statistics import mode
import sys
from termcolor import colored

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from src.results_processing.results_processing_utils.get_config import parse_json


def get_input_matrices(matrices_path, is_outer):
    """
    Finds the existing configs, test folds, and validations folds of all matrices.

    Args:
        matrices_path (string): The path of the matrices' directory.
        is_outer (bool): Whether this data is of the outer loop.

    Returns:
        dict: A dictionary of matrix-dataframes amd their prediction shapes, organized by configuration and testing fold.
    """
    
    # Gets the confusion matrix paths that were created earlier
    try:
        all_paths = os.listdir(matrices_path)
        
    except:
        print(colored("Error: could not read matrices path. Are you sure it's correct?", 'red'))
        exit(-1)
        
    # Initialization
    organized_paths = {}
    organized_shapes = {}

    # Separates by configuration
    for path in all_paths:
        configuration = path.split("/")[0].split('_')[0]
        if configuration not in organized_paths:
            organized_paths[configuration] = []
        organized_paths[configuration].append(path)

    # For each configuration, separates by testing fold
    for configuration in organized_paths:
        
        # Initialization
        config_paths = organized_paths[configuration]
        organized_paths[configuration] = {}
        organized_shapes[configuration] = {}
        
        for path in config_paths:
            filename = path.split('/')[-1]
                
            # Searchs for the test-fold name from the file name
            if not is_outer:
                
                try:
                    test_fold = re.search('_test_.*_val_', filename).captures()[0].split("_")[2] 
                    
                except:    
                    raise ValueError(colored(f"Error: No test-val pair found in the filename {filename}\n\tAre you sure it is of the inner loop?", 'red'))       
                
                if test_fold not in organized_paths[configuration]:
                    organized_paths[configuration][test_fold] = {}
                    organized_shapes[configuration][test_fold] = {}
                    
            else:
                test_fold = re.search('_test_.*_', filename).captures()[0].split("_")[2]
                
                
            # Searchs for the val-fold name from the file name, read the csv, and get shape
            if not is_outer:
                val_fold = re.search('_test_.*_val_', filename).captures()[0].split("_")[4]
                shape = re.findall(r'\d+', filename)[-1]
                organized_shapes[configuration][test_fold][val_fold] = shape
                organized_paths[configuration][test_fold][val_fold] = os.path.join(matrices_path, path)
                
            else:
                shape = int(re.search('_.*_conf_matrix.csv', filename).captures()[0].split("_")[1])
                organized_shapes[configuration][test_fold] = shape
                organized_paths[configuration][test_fold] = os.path.join(matrices_path, path)


    return organized_paths, organized_shapes



def get_matrices_of_mode_shape(shapes, matrices, is_outer):
    """
    Finds the mode length of all predictions that exist within a data folder.
    Checks if matrices should be considered in the mean value.

    Args:
        shapes (dict): The shapes (prediction rows) of the corresponding confusion matrices.
        matrices (dict): A dictionary of the matrices.
        is_outer (bool): Whether this data is of the outer loop.

    Returns:
        dict: A reduced dictionary of matrix-dataframes, organized by configuration and testing fold.
    """
    
    # Gets the mode of ALL the prediction shapes
    shapes_mode = []
    
    for configuration in matrices:
        
        for test_fold in matrices[configuration]:
            
            if not is_outer:
                for val_fold in matrices[configuration][test_fold]:
                    shapes_mode.append(shapes[configuration][test_fold][val_fold])
            else:
                shapes_mode.append(shapes[configuration][test_fold])
                
    shapes_mode = mode(shapes_mode)

    # Remove matrices whose prediction value length do not match the mode
    for configuration in matrices:
        
        for test_fold in matrices[configuration]:
            
            # Initialization
            test_fold_matrices = []
            
            # Each testing fold will have an array of coresponding validation matrices
            if not is_outer:
                
                for val_fold in matrices[configuration][test_fold]:
                    val_fold_shape = shapes[configuration][test_fold][val_fold]
                    
                    if val_fold_shape == shapes_mode:
                        test_fold_matrices.append(matrices[configuration][test_fold][val_fold])
                        
                    else:
                        print(colored(
                            f"Warning: Not including the validation fold {val_fold} in the mean of ({configuration}, {test_fold})." +
                            f"\n\tThe predictions expected to have {shapes_mode} rows, but got {val_fold_shape} rows.\n",
                            "yellow"))
                matrices[configuration][test_fold] = test_fold_matrices
                
            else:
                test_fold_shape = shapes[configuration][test_fold]
                
                if test_fold_shape == shapes_mode:
                    test_fold_matrices.append(matrices[configuration][test_fold])
                    matrices[configuration][test_fold] = test_fold_matrices
                    
                    
    return matrices



def get_mean_matrices(matrices, shapes, output_path, labels, round_to, is_outer):
    """
    This function gets the mean confusion matrix of every inner loop.

    Args:
        matrices (dict): A dictionary of matrices organized by configuration and testing fold.
        shapes (dict): A dictionary of shapes organized by configuration and testing fold.
        output_path(string): The path to write the average matrices to.
        labels(list(str)): The labels of the matrix data.
        is_outer (bool): Whether this data is of the outer loop.
    """
    
    if type(labels) == dict:
        labels = list(labels.keys())
    
    # Checks that the output folder exists.
    if not os.path.exists(output_path): os.makedirs(output_path)

    # Gets the valid matrices
    all_valid_matrices = get_matrices_of_mode_shape(shapes, matrices, is_outer)

    # The names of the rows/columns of every output matrix
    index_labels = [["Truth"] * len(labels), labels]
    columns_labels = [["Predicted"] * len(labels), labels]

    # Gets the mean of each test fold + configuration
    for configuration in matrices:
        
        for test_fold in matrices[configuration]:

            # Checks shape is the mode for each validation fold
            if is_outer:
                valid_matrices = [all_valid_matrices[configuration][t][i] for t in all_valid_matrices[configuration] for i in range(len(all_valid_matrices[configuration][t]))]
                # TODO all matrices for is_outer
                
            else:
                valid_matrices = all_valid_matrices[configuration][test_fold]
                all_matrices = matrices[configuration][test_fold]

            # Checks if length is valid for finding mean/stderr
            n_items_valid = len(valid_matrices)
            n_items_all = len(all_matrices)

            if not n_items_valid > 1:
                print(colored(f"Warning: Mean calculation skipped for testing fold {test_fold} in {configuration}."
                              + " Must have multiple validation folds.\n", 'yellow'))
                continue

            # The mean of confusion matrices and the weighted matrices
            if type(labels) == dict:
                labels = list(labels.keys())
                
            matrix_avg = pd.DataFrame(0.0, columns=labels, index=labels)
            matrix_perc_avg = pd.DataFrame(0.0, columns=labels, index=labels)
            matrix_sum_temp = pd.DataFrame(0.0, columns=labels, index=labels)
            matrix_avg.index = matrix_perc_avg.index = matrix_sum_temp.index = index_labels
            matrix_avg.columns = matrix_perc_avg.columns = matrix_sum_temp.columns = columns_labels

            # Sees if more than one item to average
            b_weight = len(all_matrices) > 1

            # Percentage matrix values
            perc_values = {}

            # Absolute version
            # Loop through every validation fold and sum the totals and weighted totals
            for val_index in range(len(valid_matrices)):
                
                val_fold = pd.read_csv(valid_matrices[val_index], index_col=[0,1], header=[0,1])
                
                for row in labels:
                    # Count the row total and add column-row items to total
                    row_total = 0
                    
                    for col in labels:
                        item = val_fold["Predicted"][col]["Truth"][row]
                        matrix_avg["Predicted"][col]["Truth"][row] += item
                        row_total += item
                   
                   
            # Percentage version
            # Loop through every validation fold and sum the totals and weighted totals
            for val_index in range(len(all_matrices)):
                val_fold = pd.read_csv(all_matrices[val_index], index_col=[0,1], header=[0,1])
                
                if b_weight:
                    perc_values[val_index] = pd.DataFrame(0, columns=labels, index=labels)
                    perc_values[val_index].index = index_labels
                    perc_values[val_index].columns = columns_labels
                    
                for row in labels:
                    # Count the row total and add column-row items to total
                    row_total = 0
                    
                    for col in labels:
                        item = val_fold["Predicted"][col]["Truth"][row]
                        matrix_sum_temp["Predicted"][col]["Truth"][row] += item
                        row_total += item

                    # Add to the weighted total
                    for col in labels:
                        item = val_fold["Predicted"][col]["Truth"][row]
                        
                        if row_total != 0:
                            weighted_item = item / row_total
                            
                        else:
                            weighted_item = 0
                            
                        perc_values[val_index]["Predicted"][col]["Truth"][row] = weighted_item
                        matrix_perc_avg["Predicted"][col]["Truth"][row] += weighted_item


            # Divides the mean-sums by the length
            for row in labels:
                for col in labels:
                    matrix_avg["Predicted"][col]["Truth"][row] /= n_items_valid
                    matrix_perc_avg["Predicted"][col]["Truth"][row] /= n_items_all

            # The standard error of the mean and weighted mean
            matrix_err = pd.DataFrame(0.0, columns=labels, index=labels)
            matrix_perc_err = pd.DataFrame(0.0, columns=labels, index=labels)
            matrix_err.index = matrix_perc_err.index = index_labels
            matrix_err.columns = matrix_perc_err.columns = columns_labels
            
            # Average
            # Sum the differences between the true and mean values squared. Divide by the num matrices minus one.
            for val_index in range(len(valid_matrices)):
                val_fold = pd.read_csv(valid_matrices[val_index], index_col=[0,1], header=[0,1])
                
                for row in labels:
                    
                    for col in labels:
                        # Sum the (true - mean) ^ 2
                        true = val_fold["Predicted"][col]["Truth"][row]
                        mean = matrix_avg["Predicted"][col]["Truth"][row]
                        matrix_err["Predicted"][col]["Truth"][row] += (true - mean) ** 2


            # Percentage
            # Sum the differences between the true and mean values squared. Divide by the num matrices minus one.
            for val_index in range(len(all_matrices)):
                val_fold = pd.read_csv(all_matrices[val_index], index_col=[0,1], header=[0,1])
                
                for row in labels:
                    
                    for col in labels:
                        # Sum the (true_weighted - mean_weighted) ^ 2
                        true_weighted = perc_values[val_index]["Predicted"][col]["Truth"][row]
                        mean_weighted = matrix_perc_avg["Predicted"][col]["Truth"][row]
                        matrix_perc_err["Predicted"][col]["Truth"][row] += (true_weighted - mean_weighted) ** 2

            # Get the standard error
            for row in labels:
                
                for col in labels:
                    
                    if n_items_valid > 1:
                        # Divide by N-1
                        matrix_err["Predicted"][col]["Truth"][row] /= n_items_valid - 1
                        # Sqrt the entire calculation
                        matrix_err["Predicted"][col]["Truth"][row] = math.sqrt(matrix_err["Predicted"][col]["Truth"][row])
                        # Divide by sqrt N
                        matrix_err["Predicted"][col]["Truth"][row] /= math.sqrt(n_items_valid)
                        
                    if n_items_all > 1:
                        # Divide by N-1
                        matrix_perc_err["Predicted"][col]["Truth"][row] /= n_items_all - 1
                        # Sqrt the entire calculation
                        matrix_perc_err["Predicted"][col]["Truth"][row] = math.sqrt(matrix_perc_err["Predicted"][col]["Truth"][row])
                        # Divide by sqrt N
                        matrix_perc_err["Predicted"][col]["Truth"][row] /= math.sqrt(n_items_all)
                    
                        
            # Creates a combination of the mean and std error
            matrix_combo = pd.DataFrame('', columns=labels, index=labels)
            matrix_perc_combo = pd.DataFrame(0, columns=labels, index=labels)
            matrix_combo.index = matrix_perc_combo.index = index_labels
            matrix_combo.columns = matrix_perc_combo.columns = columns_labels
            
            for row in labels:
                
                for col in labels:
                    matrix_combo.loc[("Truth", row), ("Predicted", col)] = \
                        f'{round(matrix_avg["Predicted"][col]["Truth"][row], round_to)} ± {round(matrix_err["Predicted"][col]["Truth"][row], round_to)}'
                    
                    if not matrix_perc_avg.empty:
                        matrix_perc_combo.loc[("Truth", row), ("Predicted", col)] = \
                            f'{round(matrix_perc_avg["Predicted"][col]["Truth"][row], round_to)} ± {round(matrix_perc_err["Predicted"][col]["Truth"][row], round_to)}'
            
            # Output all of the mean and error dataframes
            output_folder = os.path.join(output_path, f'{configuration}_{test_fold}/')
            if not os.path.exists(output_folder): os.makedirs(output_folder)
                 
            matrix_avg.round(round_to).to_csv(os.path.join(
                output_folder, f'{configuration}_{test_fold}_conf_matrix_mean.csv'))          
            matrix_err.round(round_to).to_csv(os.path.join(
                output_folder, f'{configuration}_{test_fold}_conf_matrix_stderr.csv'))
            matrix_combo.to_csv(os.path.join(
                output_folder, f'{configuration}_{test_fold}_conf_matrix_mean_stderr.csv'))

            if not matrix_perc_avg.empty:
                matrix_perc_avg.round(round_to).to_csv(os.path.join(
                    output_folder, f'{configuration}_{test_fold}_conf_matrix_perc_mean.csv'))
                matrix_perc_err.round(round_to).to_csv(os.path.join(
                    output_folder, f'{configuration}_{test_fold}_conf_matrix_perc_stderr.csv'))
                matrix_perc_combo.to_csv(os.path.join(
                    output_folder, f'{configuration}_{test_fold}_conf_matrix_perc_mean_stderr.csv'))
            print(colored(f"Mean confusion matrix results created for {test_fold} in {configuration}", 'green'))



def main():
    """
    The Main Program.
    """
    
    # Gets program configuration and run using its contents
    configuration = parse_json(os.path.abspath('src/results_processing/confusion_matrix/confusion_matrix_many_means_config.json'))

    # Reads in the matrices to average
    matrices, shapes = get_input_matrices(configuration['matrices_path'], configuration['is_outer'])

    # Averages the matrices
    get_mean_matrices(matrices, shapes, configuration['means_output_path'], configuration['label_types'], configuration['round_to'], configuration['is_outer'])



if __name__ == "__main__":
    """
    Executes Program.
    """
    
    main()
    