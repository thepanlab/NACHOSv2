import numpy as np
import os
import pandas as pd
from scipy.stats import sem as std_err
from sklearn.metrics import accuracy_score
import sys
from termcolor import colored

from src.file_processing import path_getter
from src.results_processing.results_processing_utils.get_configuration_file import parse_json


def read_data(paths_dictionary):
    """
    This will read in file-data into a config-subject dictionary.

    Args:
        paths_dictionary (dict): A dictionary of paths sorted by config and test fold.

    Returns:
        results_dictionary (dict): A dictionary of dataframes.

    Raises:
        Exception: If no data files are found within some config and test fold.
    """
    
    # Initialization
    results_dictionary = {}


    # Creates a dictionary for each config
    for configuration in paths_dictionary:
        
        # Initialization
        results_dictionary[configuration] = {}


        # Creates an array for each target-id
        for test_fold in paths_dictionary[configuration]:
            
            # Initialization
            results_dictionary[configuration][test_fold] = {}

            # Checks if the fold has files to read from
            try:
                assert paths_dictionary[configuration][test_fold], (f"Error: configuration '{configuration}' and testing fold '
                                                                    {test_fold}' had no indexed CSV files detected.
                                                                    Double-check the data files are there.")
                                        
            except AssertionError as error:
                print(colored(error, 'red'))
                sys.exit()

            # For each file, reads and stores it
            for validation_fold_path in paths_dictionary[configuration][test_fold]:
                val_fold = validation_fold_path.split("/")[-3].split("_")[-1]
                
                try:
                    results_dictionary[configuration][test_fold][val_fold] = pd.read_csv(validation_fold_path, header = None).to_numpy()
                
                except:
                    print(colored(f"Warning: {validation_fold_path} is empty.", 'yellow'))

    return results_dictionary



def get_recall_and_stderr(true_dictionary, predicted_dictionary, classes):
    """
    Gets the recall of each fold-config index.

    Args:
        true_dictionary (dict): A dictionary of true values sorted by config and test fold.
        predicted_dictionary (dict): A dictionary of predicted values sorted by config and test fold.

    Returns:
        dict: Two dictionaries of relative and real recall and standard error. Sorted by config and test fold.
    """
    
    # Gets a table of accuracy for every config
    recall = {'individual': {}, 'column': {}}
    
    for configuration in predicted_dictionary:
        
        # Initializations
        recall['individual'][configuration] = {}
        recall['column'][configuration] = {}
        
        
        # Gets the accuracy of every test-val fold pair
        for test_fold in predicted_dictionary[configuration]:
            
            # Initialization
            recall['individual'][configuration][test_fold] = {}
            
            for val_fold in predicted_dictionary[configuration][test_fold]:
                
                # Initialization
                recall['individual'][configuration][test_fold][val_fold] = {}
                
                # Stores the predicted and true values
                pred_vals = predicted_dictionary[configuration][test_fold][val_fold]
                true_vals = true_dictionary[configuration][test_fold][val_fold]
                
                # Loops through the classes
                for class_label in classes:
                    
                    # Initialization
                    recall['individual'][configuration][test_fold][val_fold][class_label] = 0
                    
                    if class_label not in recall['column'][configuration]:
                        recall['column'][configuration][class_label] = []
                    
                    # Finds where this class is in the new labels
                    label_indexes = list(np.where(true_vals == classes[class_label])[0])
                    
                    # Makes arrays of true and predicted values for this class label
                    pred_class_vals = [int(pred_vals[i]) for i in label_indexes]
                    true_class_vals = [int(true_vals[i]) for i in label_indexes]
                    
                    # Gets the accuracy of these class-values
                    recall['individual'][configuration][test_fold][val_fold][class_label] = accuracy_score(y_true=true_class_vals, y_pred=pred_class_vals)                
                                                          
                    # Total class-label accuracy mean-sum
                    recall['column'][configuration][class_label] += [recall['individual'][configuration][test_fold][val_fold][class_label]]
                
                
        # Gets the average column recall
        for class_label in recall['column'][configuration]:
            denom = len(recall['column'][configuration][class_label])

            recall['column'][configuration][class_label] = sum(recall['column'][configuration][class_label]) / denom


    # Calculates the standard error of the recall
    
    # Initialization
    standard_errors_dictionary = {}
    
    for configuration in predicted_dictionary:
        
        # Initialization
        standard_errors_dictionary[configuration] = {}
        
        # Gets the standard error of the test-fold recall
        for test_fold in predicted_dictionary[configuration]:
            standard_errors_dictionary[configuration][test_fold] = {}
            
            for val_fold in predicted_dictionary[configuration][test_fold]:
                standard_errors_dictionary[configuration][test_fold][val_fold] = {}
                
                for class_label in recall['individual'][configuration][test_fold][val_fold]:
                    standard_errors_dictionary[configuration][test_fold][val_fold][class_label]  = std_err(recall['individual'][configuration][test_fold][val_fold][class_label])
                
    # Returns accuracy
    return recall, standard_errors_dictionary



def total_output(recall, standard_error, classes, output_path, output_file, round_to, is_outer):
    """
    Produces a table of recall and standard errors by config and fold.

    Args:
        recall (dict): A dictionary of recall sorted by config and test fold.
        standard_error (dict): A dictionary of errors sorted by config and test fold.
        output_path (str): A string of the directory the output CSV should be written to.
        output_file (str): Name prefix of the output files.
        round_to (int): Gives the max numerical digits to round a value to.
        is_outer (bool): If the data is from the outer loop.
    """
    
    # Gets names of columns and subjects
    configs = list(recall['individual'].keys())
    test_folds = list(recall['individual'][configs[0]].keys())
    val_folds = list(recall['individual'][configs[0]][test_folds[0]].keys())
    
    test_folds.sort()
    val_folds.sort()
    
    # Gets output path
    if is_outer:
        output_path = os.path.join(output_path, 'outer_loop')
        
    else:
        output_path = os.path.join(output_path, 'inner_loop')
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    
    # Creates a table for each config
    for configuration in configs:
        df = pd.DataFrame(columns=['Test Fold', 'Validation Fold'] + [c for c in classes])
        
        # For every test and validation combo, gets class recall
        for test_fold in test_folds:
                 
            for val_fold in recall['individual'][configuration][test_fold]:
                row = { 'Test Fold': test_fold, 'Validation Fold': val_fold}
                
                # Adds accuracy for every class
                for class_label in classes:
                    if class_label in recall['individual'][configuration][test_fold][val_fold]:
                        row[class_label] = round(recall['individual'][configuration][test_fold][val_fold][class_label], round_to)
                    else:
                        row[class_label] = -1
            
                # Append the row to the dataframe
                df = df.append(pd.DataFrame(row, index=[0]).loc[0])
            
        # Add a row of the mean recall
        row = {'Test Fold': 'Average', 'Validation Fold': None}
        
        for class_label in classes:
            
            if class_label in recall['column'][configuration]:
                row[class_label] = round(recall['column'][configuration][class_label], round_to)
                
            else:
                row[class_label] = -1
                
        df = df.append(pd.DataFrame(row, index=[0]).loc[0])
            
        # Creates and save the Pandas dataframe
        if is_outer:
            df.to_csv(f'{output_path}/{output_file}_{configuration}_outer.csv')
            
        else:
            df.to_csv(f'{output_path}/{output_file}_{configuration}_inner.csv')



def main(configuration = None):
    """
    The main program.

    Args:
        configuration (dict): A JSON configuration as a dictionary. (Optional)
    """
    
    # Obtains a dictionary of configurations
    if configuration is None:
        configuration = parse_json('./results_processing/class_accuracy/class_accuracy_config.json')
    
    # Gets the necessary input files
    true_paths = path_getter.get_subfolder_files(configuration['data_path'], "true_label", isIndex = True, getValidation = True, isOuter = configuration['is_outer'])
    pred_paths = path_getter.get_subfolder_files(configuration['data_path'], "prediction", isIndex = True, getValidation = True, isOuter = configuration['is_outer'])

    # Reads in each file into a dictionary
    true_dictionary = read_data(true_paths)
    pred = read_data(pred_paths)
    
    # Gets recall and standard error
    recalls, stderr = get_recall_and_stderr(true_dictionary, pred, configuration['classes'])

    # Outputs results
    total_output(recalls, stderr, configuration['classes'], configuration['output_path'], configuration['output_filename'], configuration['round_to'], is_outer=False)
    print(colored("Finished writing the class recall.", 'green'))


if __name__ == "__main__":
    """
    Executes the program.
    """
    
    main()
