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
from nachosv2.setup.utils_processing import is_metric_allowed
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

    Args:
        filepath (Path): Path to the CSV file containing the training history.
        df (pd.DataFrame): DataFrame to which the processed results will be appended.
        is_cv_loop (bool): Indicates whether cross-validation (CV) is being used.
        metrics_filepath (Path): Path to the CSV file containing model evaluation metrics.

    Returns:
        pd.DataFrame: The updated DataFrame with the extracted results.
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
            
            # Run the query once
            filtered = metrics_df.query(query)
            
            # if for given test, val, and hp config
            # there are no values in metrics return the same datafram
            # Check if the metric column has at least one value
            if filtered[metric].empty:
                return df
            
            metric_value = filtered[metric].item()

            if metric == "validation_accuracy":
                accuracy_difference = abs(best_val_loss_data['validation_accuracy'].item() - metric_value)
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

        for metric in metric_columns:
            query = "test_fold==@file_info['test_fold'] and hp_config==@file_info['hp_config']"
            dict_row[f"test_{metric}"] = metrics_df.query(query)[metric]

        new_row_df = pd.DataFrame(dict_row)
        df = pd.concat([df, new_row_df], ignore_index=True)

    return df


def generate_summary(results_path: Path,
                     output_path: Optional[Path],
                     is_cv_loop: Optional[bool],
                     metrics_filepath: Optional[Path]) -> Path:
    """
    Generates a summary CSV file by combining training history and evaluation metrics
    from multiple training runs.

    This function aggregates training data (such as loss, accuracy, and epochs) from 
    individual history CSV files and merges them with optional evaluation metrics. 
    The final combined summary is saved to disk.

    Args:
        results_path (Path): Path to the directory containing training history files.
        output_path (Optional[Path]): Destination directory to save the summary file. 
            If None, defaults to a folder under `CV` or `CT`.
        is_cv_loop (Optional[bool]): Indicates whether the experiment used cross-validation loop.
        metrics_filepath (Optional[Path]): Path to a CSV file containing model evaluation metrics. 
            If provided, metrics will be merged into the summary.

    Returns:
        Path: Path to the generated summary CSV file.
    """
    
    # Find all training history files per experiment run
    history_filepath_list = get_filepath_list(results_path,
                                              "history",
                                              is_cv_loop)

    # Initialize an empty DataFrame to accumulate all history info
    df_results = pd.DataFrame()

    # Extract and print test and validation fold numbers
    # Process each history file and update the result DataFrame
    for history_path in history_filepath_list:
        df_results = fill_dataframe(history_path,
                                    df_results,
                                    is_cv_loop,
                                    metrics_filepath)

    # Construct the output file path for the summary CSV
    summary_filepath = get_filepath_from_results_path(
        results_path=results_path,
        folder_name="summary_analysis",
        file_name="results_all.csv",
        is_cv_loop=is_cv_loop,
        output_path=output_path
        )

    # Save the aggregated results to CSV
    df_results.to_csv(summary_filepath)

    return summary_filepath


def generate_avg_summary(summary_filepath: Path,
                         results_path: Path,
                         is_cv_loop: bool,
                         output_path: Optional[Path]) -> Path:
    """
    Generates a CSV file containing the average metrics for each test fold, hyperparameter 
    configuration and/or (validation fold) from a summary CSV file.

    This function reads a detailed results summary (typically one row per training run), groups 
    the results by test fold and hyperparameter configuration, calculates the average of metric 
    columns, and saves the output to a new CSV.

    Args:
        summary_csv_path (Path): Path to the summary CSV file (e.g., "results_all.csv").
        results_root_path (Path): Base directory where results are stored.
        use_cross_validation (bool): Flag indicating if results are from a cross-validation experiment.
        output_dir (Optional[Path]): Custom path to save the averaged summary. If None, defaults to `results_root_path`.

    Returns:
        Path: Path to the generated CSV file containing the averaged metrics.
    """
    # Load the summary DataFrame
    df_summary = pd.read_csv(summary_filepath, index_col=0)
    
    # Determine the starting index of metric columns based on whether CV is used
    metric_start_col_idx = 3 if is_cv_loop else 2
    metric_columns = df_summary .columns[metric_start_col_idx:].tolist()

    # Group the data by 'test_fold' and 'hp_config' and calculate the mean
    # of each metric
    df_avg = df_summary .groupby(['test_fold', 'hp_config'])[metric_columns].mean().reset_index()

    # Columns to keep unchanged
    exclude_cols = ['test_fold', 'hp_config']

    # Rename metric columns by prefixing with 'avg_'
    df_avg.rename(columns={
        col: f"avg_{col}" for col in df_avg.columns if col not in exclude_cols
    }, inplace=True)
    
    # Construct the output file path for the averaged CSV
    avg_summary_path = get_filepath_from_results_path(
        results_path=results_path,
        folder_name="summary_analysis",
        file_name="avg_per_test_fold.csv",
        is_cv_loop=True,
        output_path=output_path
        )

    df_avg.to_csv(avg_summary_path)

    return avg_summary_path


def get_best_hp_for_test_fold(avg_filepath: Path,
                              results_path: Path,
                              output_path: Optional[Path],
                              metric: str = "avg_validation_accuracy") -> Path:
    """
    Identifies the best hyperparameter configuration for each test fold based on a specified metric.

    This function reads a CSV containing average performance metrics per fold and hyperparameter config,
    then selects the configuration with the highest score for each fold based on the target metric.

    Args:
        avg_filepath (Path): Path to the CSV file containing average metrics for each fold/config.
        results_path (Path): Root path where experiment results are stored.
        output_path (Optional[Path]): Optional directory to save the output CSV. Defaults to a folder under `results_path`.
        metric (str): The metric to use for selecting the best configuration (default is "avg_validation_accuracy").

    Returns:
        Path: Path to the generated CSV file containing the best hyperparameter config per test fold.
    """

    # Load the DataFrame from the averaged metrics CSV
    df_avg_metrics = pd.read_csv(avg_filepath, index_col=0)
    
    # TODO: test for multiple hp configurations
    # Select the row with the highest value of the target metric for each test fold
    best_config_per_test_fold = df_avg_metrics.loc[
        df_avg_metrics.groupby('test_fold')[metric].idxmax()
    ]
    
    # Build the output path for the CSV storing the best configurations
    best_config_output_path = get_filepath_from_results_path(
        results_path=results_path,
        folder_name="best_analysis",
        file_name="best_hp_for_test.csv",
        is_cv_loop=True,
        output_path=output_path
    )

    best_config_per_test_fold.to_csv(best_config_output_path)

    return best_config_output_path


def generate_ct_hp_configurations(best_filepath: Path,
                                  results_path: Path,
                                  configuration_cv_path: Path):
    """
    Generates cross-testing (CT) training and hyperparameter configuration files.

    This function reads the best configuration metrics from a CSV file (best_filepath) 
    and if hyperparameter random search was used. The best hyperparameter configuration is selected
    For each entry in the best configurations, it generates a YAML file containing a training configuration 
    and another YAML file with the corresponding hyperparameter configuration for cross-testing.
    
    Parameters:
        best_filepath (Path): Path to the CSV file containing the best 
                              configuration metrics, including test fold, 
                              hyperparameter configuration index, and average 
                              number of epochs.
        results_path (Path): Root directory where result directories (e.g.,
                             CT configurations and CV HPO configurations) are located.
        configuration_cv_path (Path): Path to the cross-validation configuration file,
                                      used to retrieve initial settings 
                                      (including the "use_hpo" flag).

    Returns:
        None
        (The function generates configuration YAML files as a side effect.)
    """
    ensure_path_exists(configuration_cv_path)
    use_hpo = get_config(configuration_cv_path)["use_hpo"]

    ct_configuration_path = results_path / "CT" / "configurations"
    # make directory from path
    ct_configuration_path.mkdir(parents=True, exist_ok=True)
    
    df_best = pd.read_csv(best_filepath, index_col=0)
    
    if use_hpo:
        path_hp_configurations = results_path / "CV" / \
            "hp_random_search" / "hp_configurations.csv"
                
        df_hp_random_search = pd.read_csv(path_hp_configurations, index_col=0)

    for row in df_best.itertuples():
        test_fold = row.test_fold
        hp_config_index = row.hp_config
        n_epochs = round(row.avg_n_epochs)

        ct_training_config_path = ct_configuration_path / \
            f"training_config_test_{test_fold}_hpconfig_{hp_config_index}.yml"
        ct_hp_config_path = ct_configuration_path / \
            f"hp_config_test_{test_fold}_hpconfig_{hp_config_index}.yml"

        training_config_dict = get_config(configuration_cv_path)

        training_config_dict["configuration_filepath"] = str(ct_hp_config_path)
        training_config_dict["test_fold_list"] = test_fold
        # empty validation list
        training_config_dict["validation_fold_list"] = []
        # use_hpo to False
        training_config_dict["use_hpo"] = False
        # save yml file
        save_dict_to_yaml(training_config_dict, ct_training_config_path)

        if use_hpo:
            # Based on the hp_config_index, load the corresponding hyperparameter configuration
            df_best_hp_config = df_hp_random_search.query("hp_config_index==@hp_config_index")
            hp_config_dict = df_best_hp_config.to_dict(orient='records')[0]
        else:
            hp_config_dict = get_config(training_config_dict["configuration_filepath"])
            
            del training_config_dict["validation_fold_list"]
            hp_config_dict.pop("n_combinations", None)
            del hp_config_dict["patience"]
            
        hp_config_dict["n_epochs"] = n_epochs
        save_dict_to_yaml(hp_config_dict, ct_hp_config_path)


def main():
    """
    Main entry point of the program.

    This function arranges the entire model evaluation pipeline:
    - Parses configuration and command-line arguments
    - Extracts metrics from prediction results
    - Generates a summary of training history
    - If CV loop:
        - Computes average metrics across CV folds
        - Selects the best hyperparameters
        - Prepares Cross-testing configurations
    """
    # ------------------------------
    # Step 1: Load and parse arguments
    # ------------------------------
    args = parse_command_line_args()
    config_dict = get_config(args['file'])

    # ------------------------------
    # Step 2: Resolve paths and flags
    # ------------------------------
    results_path = Path(config_dict['results_path'])
    ensure_path_exists(results_path)

    output_path = config_dict.get('output_path', None)
    if output_path is not None:
        output_path = Path(output_path)

    is_cv_loop = config_dict.get('is_cv_loop', None)
    
    # Determine the metric used for selecting the best hyperparameter configuration.
    # If a specific metric (e.g., "f1_micro") is provided, format it as "avg_validation_f1_micro".
    raw_metric = config_dict.get('metric_for_selection', 'accuracy').strip()
    metric_for_selection = f"avg_validation_{raw_metric}"  # always formatted consistently
    if not is_metric_allowed(raw_metric):
        raise ValueError(f"{raw_metric} is not allowed in list of metrics.")
    # ------------------------------
    # Step 3: Define list of metrics to extract
    # ------------------------------
    metrics_list = ["accuracy"] # Default metric
    metrics_list_config = config_dict.get('metrics_list', None)
    
    if isinstance(metrics_list_config, list): 
        metrics_list.extend(metrics_list_config)
    elif isinstance(metrics_list_config, str):
        if metrics_list_config != "accuracy":
            metrics_list.append(metrics_list_config)


    # ------------------------------
    # Step 4: Generate metrics CSV file
    # ------------------------------
    metrics_filepath = generate_metrics_file(
        metrics_list=metrics_list,
        results_path=results_path,
        output_path=output_path,
        is_cv_loop=is_cv_loop)

    # ------------------------------
    # Step 5: Generate training summary CSV
    # ------------------------------
    summary_filepath = generate_summary(results_path,
                                        output_path,
                                        is_cv_loop,
                                        metrics_filepath)
    
    # ------------------------------
    # Step 6: If CV loop, compute average metrics and best HP
    # ------------------------------
    if is_cv_loop:
        avg_summary_filepath = generate_avg_summary(summary_filepath,
                                                    results_path,
                                                    is_cv_loop,
                                                    output_path)

        configuration_cv_path = config_dict.get('configuration_cv_path', None)

        # Get path to CV configuration folder
        if configuration_cv_path is None:
            raise ValueError("Configuration path('configuration_path')"
                             " must be provided for cross-validation loop.")
       
        # ------------------------------
        # Step 7: Identify best hyperparameters
        # ------------------------------
        best_summary_filepath = get_best_hp_for_test_fold(
            avg_filepath=avg_summary_filepath,
            results_path=results_path,
            output_path=output_path,
            metric=metric_for_selection)

        # ------------------------------
        # Step 8: Generate final CT configurations from best CV results
        # ------------------------------
        generate_ct_hp_configurations(
            best_filepath=best_summary_filepath,
            results_path=results_path,
            configuration_cv_path=configuration_cv_path)


if __name__ == "__main__":
    main()
