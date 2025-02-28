from pathlib import Path
import re
from termcolor import colored
from sklearn import metrics
import pandas as pd
from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.get_config import get_config
from nachosv2.setup.get_filepath_list import get_filepath_list
from nachosv2.setup.get_other_result import get_other_result


def generate_individual_confusion_matrix(path: Path) -> pd.DataFrame:
    
    df_results = pd.read_csv(path)
    actual = df_results['true_label']
    predicted = df_results['predicted_class']
    
    confusion_matrix = metrics.confusion_matrix(actual, predicted) 

    # get class_names file
    df_class_names = get_other_result(path, "class_names")
    
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


def get_default_folder(path: Path):
    index_training_results = path.parts.index("training_results")
    default_folder_path = Path(*path.parts[:index_training_results]) / "confusion_matrix"
    
    return default_folder_path


def get_filepath_confusion_matrix(config:dict,
                                  path: Path):
    """
    Get the name of the confusion matrix file.

    Args:
        path (Path): The path of the confusion matrix file.

    Returns:
        string: The name of the confusion matrix file.
    """
    
    file_name = path.name.replace("prediction_results", "confusion_matrix")
    if  "output_folder_path" not in config:
        folder_path = get_default_folder(path)
        print(colored(f"\nUsing default folder: {folder_path}.", 'magenta'))
    else:
        folder_path = Path(config["output_folder_path"])
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
    config_dict = get_config(args['file'])

    # Look for csv files with specific format
    if not Path(config_dict['results_path']).exists():
        raise FileNotFoundError(print(colored(f"Path {config_dict['results_path']} does not exist.", "red")))

    results_path = Path(config_dict['results_path'])
    
    suffix_filename = "prediction_results"
    predictions_file_path_list = get_filepath_list(results_path,
                                                   suffix_filename)
    
    # Regex to match test and validation info
    pattern = r"test_([A-Za-z0-9]+)_hpconfig_([0-9])+_val_([A-Za-z0-9]+)"

    # Extract and print test and validation fold numbers
    for path in predictions_file_path_list:
        filename = path.name
        match = re.search(pattern, filename)
        if match:
            test_fold, hp_config, val_fold = match.groups()
            print(f"File: {filename}")
            print(f"Test fold: {test_fold}")
            print(f"Hyperparameter configuration index: {hp_config}")
            print(f"Validation fold: {val_fold}")
            print("-----")
            
        # TODO: get class name files by modifying filename
        # 
        cf_df = generate_individual_confusion_matrix(path)

        cf_filepath = get_filepath_confusion_matrix(config_dict, path)
        cf_df.to_csv(cf_filepath)
        
        # Add column to specify the predicted and the ground truth    
        # TODO: make confusion matrix for all the validation
        # verify that all have the same amount of files


if __name__ == "__main__":
    main()
