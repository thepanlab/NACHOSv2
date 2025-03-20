from pathlib import Path
from typing import Optional
from termcolor import colored
import pandas as pd
from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.utils import get_filepath_list
from nachosv2.setup.utils import get_filepath_from_results_path
from nachosv2.setup.files_check import ensure_path_exists
from nachosv2.setup.get_config import get_config
from nachosv2.setup.utils_processing import save_dict_to_yaml
from nachosv2.setup.utils_processing import parse_filename
from nachosv2.results_processing.get_metrics.get_metrics import (
    generate_metrics_file
)

def fill_dataframe(filepath: Path,
                   df: pd.DataFrame,
                   is_cv_loop: bool,
                   metrics_filepath: Path) -> pd.DataFrame:
    """
    Processes a CSV file containing model training history, extracts relevant metrics, 
    and appends a summarized row to an existing DataFrame.

    Parameters:
    ----------
    filepath : Path
        Path to the CSV file containing the training history.
    df : pd.DataFrame
        DataFrame to which the processed results will be appended.
    is_cv_loop : bool
        Indicates whether cross-validation (CV) is being used.
    metrics_filepath : Path
        Path to the CSV file containing model evaluation metrics.

    Returns:
    -------
    pd.DataFrame
        The updated DataFrame with the extracted results.
    """
    # Read CSV file into DataFrame
    history_df = pd.read_csv(filepath, index_col=0)
    metrics_df = pd.read_csv(metrics_filepath, index_col=0)

    # if CV loop, 3 first columns are test, hp_config, and val
    # if CT loop, 2 first columns are test, val    
    index_col_metric = 3 if is_cv_loop else 2
    metric_columns = metrics_df.columns[index_col_metric:]
    
    # Cross-validation loop
    if is_cv_loop:
        # Example: Extracting the epoch with the lowest loss (assuming a column 'loss' exists)
        # You need to adjust this based on your actual CSV column names and what you define as 'best'
        # Logic: it identifies the minimum value for 'validation loss'
        best_val_loss_index = history_df['validation_loss'].idxmin()
        best_val_loss_data = history_df.iloc[best_val_loss_index]

        # Extract information from the file name
        file_info = parse_filename(filepath, is_cv_loop)

        dict_row = {
            'test_fold': file_info['test_fold'],
            'hp_config': file_info['hp_config'],
            'val_fold': file_info['val_fold'],
            # since the index is 0-based, we add 1 to get the actual epoch number
            'n_epochs': [best_val_loss_index+1],
            'training_loss': [best_val_loss_data['training_loss']],
            'validation_loss': [best_val_loss_data['validation_loss']],
            'training_accuracy': [best_val_loss_data['training_accuracy']],
            'validation_accuracy': [best_val_loss_data['validation_accuracy']]
        }

        for metric in metric_columns:
            query = (
                "test_fold==@file_info['test_fold'] and "
                "val_fold==@file_info['val_fold'] and "
                "hp_config==@file_info['hp_config']"
                )
            metric_value = metrics_df.query(query)[metric].item()

            if metric == "validation_accuracy":
                accuracy_difference = abs(best_val_loss_data['validation_accuracy'].item() - metric_value)
                if metric == "accuracy":
                    if accuracy_difference > 0.02:
                        raise ValueError("Validation accuracy difference is greater than 2% for history and predictions")

            dict_row[metric] = metrics_df.query(query)[metric]

        # Prepare row as a DataFrame to be concatenated
        new_row_df = pd.DataFrame(dict_row)

        # Concatenate the new row to the existing DataFrame
        df = pd.concat([df, new_row_df], ignore_index=True)

    # Cross-testing loop
    else:
        # Extract information from the file name
        file_info = parse_filename(filepath, is_cv_loop)

        # Prepare row as a DataFrame to be concatenated
        dict_row = {
            'test_fold': file_info['test_fold'],
            'hp_config': file_info['hp_config'],
            # since the index is 0-based, we add 1 to get the actual epoch number
            'n_epochs': [history_df.iloc[-1].name+1], # it extracts 0-based indexed of last row
            'training_loss': [history_df.iloc[-1]['training_loss']],
            'training_accuracy': [history_df.iloc[-1]['training_accuracy']],
        }

        for metric in l_metrics:
            query = "test_fold==@file_info['test_fold'] and hp_config==@file_info['hp_config']"
            dict_row[f"test_{metric}"] = metrics_df.query(query)[metric]

        new_row_df = pd.DataFrame(dict_row)
        df = pd.concat([df, new_row_df], ignore_index=True)

    return df


def generate_summary(results_path: Path,
                     output_path: Optional[Path],
                     is_cv_loop: Optional[bool],
                     metrics_filepath: Optional[Path]) -> Path:
    
    history_filepath_list = get_filepath_list(results_path,
                                              "history",
                                              is_cv_loop)
    
    df_results = pd.DataFrame()
    
    # Extract and print test and validation fold numbers
    for history_path in history_filepath_list:
        df_results = fill_dataframe(history_path,
                                    df_results,
                                    is_cv_loop,
                                    metrics_filepath)
    
    summary_filepath = get_filepath_from_results_path(results_path=results_path,
                                                      folder_name="summary_analysis",
                                                      file_name="results_all.csv",
                                                      is_cv_loop=is_cv_loop,
                                                      output_path=output_path)

    df_results.to_csv(summary_filepath)

    return summary_filepath


def generate_avg_summary(summary_filepath: Path,
                         results_path: Path,
                         is_cv_loop: bool,
                         output_path: Optional[Path]) -> Path:
    
    data = pd.read_csv(summary_filepath, index_col=0)  # Change to the path of your CSV file
    index_start_column = 3 if is_cv_loop else 2
    
    columns_to_average = data.columns[index_start_column:].tolist()

    # Group the data by 'test_fold' and 'hp_config' and calculate the mean
    grouped_data = data.groupby(['test_fold', 'hp_config'])[columns_to_average].mean().reset_index()

    # Rename columns to reflect they are averages
    grouped_data = grouped_data.rename(columns={
        'n_epochs': 'avg_n_epochs',
        'training_loss': 'avg_training_loss',
        'validation_loss': 'avg_validation_loss',
        'training_accuracy': 'avg_training_accuracy',
        'validation_accuracy': 'avg_validation_accuracy'
    })
    
    average_filepath = get_filepath_from_results_path(results_path=results_path,
                                                      folder_name="summary_analysis",
                                                      file_name="avg_for_test.csv",
                                                      is_cv_loop=True,
                                                      output_path=output_path)

    grouped_data.to_csv(average_filepath)

    return average_filepath


def get_best_hp_for_test(avg_filepath: Path,
                         results_path: Path,
                         output_path: Optional[Path],
                         metric: str = "avg_validation_accuracy",
                         use_hpo: bool = False) -> Path:
    
    data = pd.read_csv(avg_filepath, index_col=0)
    
    # TODO: test for multiple hp configurations
    best_config_by_fold = data.loc[data.groupby('test_fold')[metric].idxmax()]
    
    best_filepath = get_filepath_from_results_path(results_path=results_path,
                                                      folder_name="best_analysis",
                                                      file_name="best_hp_for_test.csv",
                                                      is_cv_loop = True,
                                                      output_path=output_path)

    best_config_by_fold.to_csv(best_filepath)

    return best_filepath


def generate_ct_hp_configurations(best_filepath: Path,
                                  results_path: Path,
                                  configuration_cv_path: Path,
                                  use_hpo=False):
    """
    Processes a single row from the best configurations file to generate CT training and HP configuration files.

    Parameters:
    ----------
    row : pd.Series
        Row from the DataFrame containing test fold and hyperparameter index.
    training_config_dict : Dict
        Training configuration dictionary.
    hp_config_dict : Dict
        Hyperparameter configuration dictionary.
    ct_configuration_path : Path
        Path to the CT configuration directory.
    """
    training_config_dict = get_config(configuration_cv_path)
    if use_hpo:
        # TODO
        pass
    else:
        hp_config_dict = get_config(training_config_dict["configuration_filepath"])

    data = pd.read_csv(best_filepath, index_col=0)

    # based on config:
    # * delete validation_fold_list
    del training_config_dict["validation_fold_list"]
    hp_config_dict.pop("n_combinations", None)
    del hp_config_dict["patience"]

    ct_configuration_path = results_path / "CT" / "configurations"
    # make directory from path
    ct_configuration_path.mkdir(parents=True, exist_ok=True)

    for row in data.itertuples():
        test_fold = row.test_fold
        hp_config_index = row.hp_config
        n_epochs = round(row.avg_n_epochs)
        # TODO
        # if use_hpo==true, then I would need to retrieve the path of the different files and extract its features
        # change test fold: to values
        training_config_path = ct_configuration_path / f"training_config_test_{test_fold}_hpconfig_{hp_config_index}.yml"
        hp_config_path = ct_configuration_path /f"hp_config_test_{test_fold}_hpconfig_{hp_config_index}.yml"

        training_config_dict["configuration_filepath"] = str(hp_config_path)
        training_config_dict["test_fold_list"] = test_fold

        # save yml file
        save_dict_to_yaml(training_config_dict, training_config_path)

        hp_config_dict["n_epochs"] = n_epochs
        save_dict_to_yaml(hp_config_dict, hp_config_path)


def main():
    """
    The main body of the program
    """
    # Parses the command line arguments
    args = parse_command_line_args()
    # Defines the arguments
    config_dict = get_config(args['file'])

    results_path = Path(config_dict['results_path'])
    ensure_path_exists(results_path)

    output_path = config_dict.get('output_path', None)
    if output_path is not None:
        output_path = Path(output_path)

    is_cv_loop = config_dict.get('is_cv_loop', None)

    metrics_list = ["accuracy"]
    metrics_list_config = config_dict.get('metrics_list', None)
    
    if isinstance(metrics_list_config, list): 
        metrics_list.extend(metrics_list_config)
    elif isinstance(metrics_list_config, str):
        if metrics_list_config != "accuracy":
            metrics_list.append(metrics_list_config)

    metrics_filepath = generate_metrics_file(
        metrics_list=metrics_list,
        results_path=results_path,
        output_path=output_path,
        is_cv_loop=is_cv_loop)

    summary_filepath = generate_summary(results_path,
                                        output_path,
                                        is_cv_loop,
                                        metrics_filepath)

    if is_cv_loop:
        avg_summary_filepath = generate_avg_summary(summary_filepath,
                                                    results_path,
                                                    is_cv_loop,
                                                    output_path)

        configuration_cv_path = config_dict.get('configuration_cv_path', None)

        if configuration_cv_path is None:
            raise ValueError("Configuration path('configuration_path')"
                             " must be provided for cross-validation loop.")

        configuration_cv_path = Path(configuration_cv_path)

        ensure_path_exists(configuration_cv_path)

        best_summary_filepath = get_best_hp_for_test(
            avg_filepath=avg_summary_filepath,
            results_path=results_path,
            output_path=output_path,
            metric="avg_validation_accuracy",
            use_hpo=False)

        #  generate configuration files for test
        generate_ct_hp_configurations(
            best_filepath=best_summary_filepath,
            results_path=results_path,
            configuration_cv_path=configuration_cv_path,
            use_hpo=False)


if __name__ == "__main__":
    main()
