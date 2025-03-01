from pathlib import Path
import re
from typing import Optional
import os
import math
from termcolor import colored
import pandas as pd
from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.get_config import get_config
from nachosv2.setup.utils import get_filepath_list
from nachosv2.setup.utils import determine_if_cv_loop
from nachosv2.setup.utils import get_filepath_from_results_path

def count_epochs(history_paths, is_outer):
    """
    Reads in the history paths and finds the epoch with the minimum validation loss.

    Args:
        history_paths (dict): A dictionary of file locations.
        is_outer (bool): If this data is from the outer loop or not.

    Raises:
        Exception: When no history files exist for some item.

    Returns:
        dict: A dictionary of minimum epoch losses.
    """
    
    # Stores output dataframes by model
    model_dfs = {}

    # Each model will have its own dictionary, a later dataframe
    for model in history_paths:
        model_dfs[model] = {}

        # Every subject is the kth test set (row), find the column values
        for row in history_paths[model]:
            row_name = row
            model_dfs[model][row_name] = {}

            # Read in the data at this path
            for path in history_paths[model][row]:
                # Check if the fold has files to read from
                if not path:
                    raise Exception(colored(f"Error: model '{model}' had no history file detected.", 'red'))

                # Reads the file for this column/subject, get number of rows (epochs)
                data = pd.read_csv(path)
                min_index = -1
                min_epoch = float("inf")
                
                if is_outer:
                    key = 'loss'
                    
                else:
                    key = 'val_loss'
                    
                for row in data[key].index:
                    
                    if data[key][row] < min_epoch:
                        min_epoch = data[key][row]
                        min_index = row

                # Adds the epoch with the lowest loss the model's dataframe 
                col_name = path.split("/")[-2].split("_")[-1]
                model_dfs[model][row_name][col_name] = min_index + 1


    return model_dfs


def save_epoch_counts(epochs, output_path, config_nums, is_outer):
    """ 
    This will save a CSV file containing the following columns:
    test_fold, config, conifg_index, val_fold, epochs

    Args:
        epochs (dict): Dictionary of epochs with minimum loss.
        output_path (str): Path to write files into.
        config_nums (dict): The configuration indexes of the data.
        is_outer (bool): If this data is from the outer loop.
    """
    
    # Creates a new dataframe to output
    col_names = ["test_fold", "config", "config_index", "val_fold", "epochs"]
    df = pd.DataFrame(columns=col_names)

    # Re-format data to match the columns above
    configs = list(epochs.keys())
    for testing_fold_index in range(len(epochs[configs[0]])):
        
        for config in configs:
            
            testing_fold = list(epochs[config].keys())[testing_fold_index]
            
            for validation_fold in epochs[config][testing_fold]:

                # Each row should contain the given columns
                df_temp = pd.DataFrame({
                            col_names[0]: [testing_fold],
                            col_names[1]: [config],
                            col_names[2]: [config_nums[config]],
                            col_names[3]: [validation_fold],
                            col_names[4]: [epochs[config][testing_fold][validation_fold]]
                            })
                
                df = pd.concat([df, df_temp], ignore_index=True)

    # Prints to file
    if is_outer:
        file_name = 'epochs_outer.csv'
        
    else:
        file_name = 'epochs_inner.csv'
        
    df = df.sort_values(by=[col_names[0], col_names[2], col_names[1]], ascending=True)
    df.to_csv(os.path.join(output_path, file_name), index=False)
    print(colored('Successfully printed epoch results to: ' + file_name, 'green'))



def save_epoch_avg_stderr(epochs, output_path, config_nums):
    """ 
    This will save CSV containing average epoch standard errors.
    It has the columns: test_fold, config, config-index, avg_epochs, std_err
    
    Args:
        epochs (dict): Dictionary of epochs with minimum loss.
        output_path (str): Path to write files into.
        config_nums (dict): The configuration indexes of the data.
    """
    
    # Creates a new dataframe to output
    col_names = ["test_fold", "config", "config-index", "avg_epochs", "std_err"]
    df = pd.DataFrame(columns=col_names)

    # Re-formats data to match the columns above
    for config in epochs:
        
        for test_fold in epochs[config]:

            # Count epochs, get mean
            epoch_mean = 0
            n_val_folds = len(epochs[config][test_fold])
            
            for validation_fold in epochs[config][test_fold]:
                epoch_mean += epochs[config][test_fold][validation_fold]
                
            epoch_mean = epoch_mean / n_val_folds

            # Gets standard deviation
            stdev = 0
            
            for validation_fold in epochs[config][test_fold]:
                stdev += (epochs[config][test_fold][validation_fold] - epoch_mean) ** 2
                
            stdev = math.sqrt(stdev / (n_val_folds - 1))

            # Each row should contain the given columns
            df_temp = pd.DataFrame({
                col_names[0]: [test_fold],
                col_names[1]: [config],
                col_names[2]: [config_nums[config]],
                col_names[3]: [epoch_mean],
                col_names[4]: [stdev / math.sqrt(n_val_folds)]
            })
            
            df = pd.concat([df, df_temp], ignore_index=True)

    # Print to file
    file_name = 'epoch_inner_avg_stderr.csv'
    df = df.sort_values(by=[col_names[0], col_names[1]], ascending=True)
    df.to_csv(os.path.join(output_path, file_name), index=False)
    print(colored('Successfully printed epoch averages/stderrs to: ' + file_name, 'green'))


def fill_dataframe(filepath: Path,
                   df: pd.DataFrame,
                   is_cv_loop: bool):
    
    # Read CSV file into DataFrame
    history_df = pd.read_csv(filepath)

    # Example: Extracting the epoch with the lowest loss (assuming a column 'loss' exists)
    # You need to adjust this based on your actual CSV column names and what you define as 'best'
    if 'validation_loss' in history_df.columns:
        # Logic: it identifies the minimum value for 'validation loss'
        best_epoch_index = history_df['validation_loss'].idxmin()
        best_epoch_data = history_df.iloc[best_epoch_index]

         # Extract information from the file name
        parts = filepath.stem.split('_')
        test_fold = parts[1]  # Adjust based on actual index in your filenames
        hp_config = parts[3]  # Adjust based on actual index in your filenames
        val_fold = parts[5] if is_cv_loop else None  # Check if it's a CV loop

        # Prepare row as a DataFrame to be concatenated
        new_row_df = pd.DataFrame({
            'test_fold': [test_fold],
            'hp_config': [hp_config],
            'val_fold': [val_fold],
            'best_epoch': [best_epoch_index],
            'best_loss': [best_epoch_data['train_loss']]
        })

        # Concatenate the new row to the existing DataFrame
        df = pd.concat([df, new_row_df], ignore_index=True)

    return df
    

def generate_summary_epochs(results_path: Path,
                            output_path: Optional[Path],
                            is_cv_loop: Optional[bool]=None):
    
    suffix_filename = "history"
    history_filepath_list = get_filepath_list(results_path,
                                              suffix_filename)
    
    if is_cv_loop is None:
        is_cv_loop = determine_if_cv_loop(history_filepath_list[0])
    
    if is_cv_loop:
        # Regex to match test and validation info
        # cross-validation
        pattern = r"test_([A-Za-z0-9]+)_hpconfig_([0-9]+)_val_([A-Za-z0-9]+)"
    # cross-validation
    else:
        pattern = r"test_([A-Za-z0-9]+)_hpconfig_([0-9]+)"

    df_epochs = pd.DataFrame()
    
    # Extract and print test and validation fold numbers
    for history_path in history_filepath_list:
        filename = history_path.name
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            test_fold = groups[0]
            hp_config = groups[1]
            val_fold = groups[2] if len(groups) > 2 else None  # Conditional assignment for val_fold
            
            print(f"File: {filename}")
            print(f"Test fold: {test_fold}")
            print(f"Hyperparameter configuration index: {hp_config}")
            if val_fold:  # Print validation fold only if it's available
                print(f"Validation fold: {val_fold}")
            print("-----")
            
        df_epochs = fill_dataframe(history_path, df_epochs, is_cv_loop)

        print("hello")
    
    epochs_filepath = get_filepath_from_results_path(results_path,
                                                    "epochs_analysis",
                                                    "epochs_results_all.csv",
                                                     output_path)
    
    df_epochs.to_csv(epochs_filepath)


def main(config = None):
    """
    The main body of the program
    """
    # TODO
    # modify to new values
    # Obtain a dictionary of configurations
    
        # Parses the command line arguments
    args = parse_command_line_args()
    
    # Defines the arguments
    config_dict = get_config(args['file'])
    
    if not Path(config_dict['results_path']).exists():
        raise FileNotFoundError(print(colored(f"Path {config_dict['results_path']} does not exist.", "red")))
    
    results_path = Path(config_dict['results_path'])
    
    output_path = config_dict.get('output_path', None)
    if output_path is not None:
        output_path = Path(output_path)
    
    is_cv_loop = config_dict.get('is_cv_loop', None)
    
    generate_summary_epochs(results_path, output_path,
                            is_cv_loop)
    

if __name__ == "__main__":
    
    main()
