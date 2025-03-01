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
from nachosv2.setup.files_check import ensure_path_exists

# def save_epoch_avg_stderr(epochs, output_path, config_nums):
#     """ 
#     This will save CSV containing average epoch standard errors.
#     It has the columns: test_fold, config, config-index, avg_epochs, std_err
    
#     Args:
#         epochs (dict): Dictionary of epochs with minimum loss.
#         output_path (str): Path to write files into.
#         config_nums (dict): The configuration indexes of the data.
#     """
    
#     # Creates a new dataframe to output
#     col_names = ["test_fold", "config", "config-index", "avg_epochs", "std_err"]
#     df = pd.DataFrame(columns=col_names)

#     # Re-formats data to match the columns above
#     for config in epochs:
        
#         for test_fold in epochs[config]:

#             # Count epochs, get mean
#             epoch_mean = 0
#             n_val_folds = len(epochs[config][test_fold])
            
#             for validation_fold in epochs[config][test_fold]:
#                 epoch_mean += epochs[config][test_fold][validation_fold]
                
#             epoch_mean = epoch_mean / n_val_folds

#             # Gets standard deviation
#             stdev = 0
            
#             for validation_fold in epochs[config][test_fold]:
#                 stdev += (epochs[config][test_fold][validation_fold] - epoch_mean) ** 2
                
#             stdev = math.sqrt(stdev / (n_val_folds - 1))

#             # Each row should contain the given columns
#             df_temp = pd.DataFrame({
#                 col_names[0]: [test_fold],
#                 col_names[1]: [config],
#                 col_names[2]: [config_nums[config]],
#                 col_names[3]: [epoch_mean],
#                 col_names[4]: [stdev / math.sqrt(n_val_folds)]
#             })
            
#             df = pd.concat([df, df_temp], ignore_index=True)

#     # Print to file
#     file_name = 'epoch_inner_avg_stderr.csv'
#     df = df.sort_values(by=[col_names[0], col_names[1]], ascending=True)
#     df.to_csv(os.path.join(output_path, file_name), index=False)
#     print(colored('Successfully printed epoch averages/stderrs to: ' + file_name, 'green'))


def parse_filename(filepath: Path,
                   is_cv_loop: bool):
    """
    Parse parts from the filename to extract test fold, hyperparameter configuration, and validation fold.
    """
    parts = filepath.stem.split('_')
    return {
        'test_fold': parts[1],  # Assumes specific filename structure
        'hp_config': parts[3],
        'val_fold': parts[5] if is_cv_loop else None
    }


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
        file_info = parse_filename(filepath, is_cv_loop)

        # Prepare row as a DataFrame to be concatenated
        new_row_df = pd.DataFrame({
            'test_fold': file_info['test_fold'],
            'hp_config': file_info['hp_config'],
            'val_fold': file_info['val_fold'],
            'best_epoch': [best_epoch_index],
            'best_loss': [best_epoch_data['train_loss']]
        })

        # Concatenate the new row to the existing DataFrame
        df = pd.concat([df, new_row_df], ignore_index=True)

    return df
    

def generate_summary_epochs(results_path: Path,
                            output_path: Optional[Path],
                            is_cv_loop: Optional[bool]=None):
    
    history_filepath_list = get_filepath_list(results_path,
                                              "history")
    
    df_epochs = pd.DataFrame()
    
    # Extract and print test and validation fold numbers
    for history_path in history_filepath_list:
        filename = history_path.name
            
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
    # Parses the command line arguments
    args = parse_command_line_args()
    # Defines the arguments
    config_dict = get_config(args['file'])
    ensure_path_exists(config_dict['results_path'])
    results_path = Path(config_dict['results_path'])
    output_path = config_dict.get('output_path', None)
    
    if output_path is not None:
        output_path = Path(output_path)
    
    is_cv_loop = config_dict.get('is_cv_loop', None)
    
    generate_summary_epochs(results_path, output_path,
                            is_cv_loop)
    

if __name__ == "__main__":
    
    main()
