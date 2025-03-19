from pathlib import Path
from typing import Optional, List
from termcolor import colored
from sklearn import metrics
import pandas as pd
from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.get_config import get_config
from nachosv2.setup.utils import get_other_result
from nachosv2.setup.utils import get_filepath_list
from nachosv2.setup.utils import determine_if_cv_loop
from nachosv2.setup.utils import get_default_folder
from nachosv2.setup.utils import get_newfilepath_from_predictions
from nachosv2.setup.files_check import ensure_path_exists
from nachosv2.setup.utils_processing import generate_individual_metric
from nachosv2.setup.utils_processing import parse_filename
from nachosv2.setup.utils import get_filepath_from_results_path


def generate_metrics_file(metrics_list: List[str],
                          results_path: Path,
                          is_cv_loop: bool,
                          output_path: Optional[Path]) -> Path:
    """
    Generates a CSV file containing evaluation metrics for machine learning predictions.

    Parameters:
    ----------
    metrics_list : List[str]
        A list of metric names to be extracted from each prediction results file.
    results_path : Path
        Path to the directory containing prediction results.
    is_cv_loop : bool
        Specifies whether cross-validation (CV) is being used. If True, results 
        include cross-validation results.
    output_path : Optional[Path]
        Directory where the generated metrics file should be saved. If None, 
        the default results path will be used.

    Returns:
    -------
    Path
        The file path of the generated metrics CSV.
    """
        
    # Define filename suffix for prediction result files 
    suffix_filename = "prediction_results"
    
    # Get the list of file paths containing prediction results
    predictions_path_list = get_filepath_list(
        directory_to_search_path=results_path,
        string_in_filename=suffix_filename,
        is_cv_loop=is_cv_loop)
    
    # Initialize an empty DataFrame to store metrics results
    df_results = pd.DataFrame()

    # Process each prediction file
    for predictions_path in predictions_path_list:
        # Generate a dictionary of extracted metrics
        metrics_dict = generate_individual_metric(metrics_list,
                                                  predictions_path)
        # Extract file metadata i.e. dict with
        # test fold, hyperparameter configuration index, and validation fold
        file_info = parse_filename(predictions_path, is_cv_loop)

        if not is_cv_loop:
            file_info.pop('val_fold', None)

        file_info.update(metrics_dict)
        new_row_df = pd.DataFrame(file_info)

        # Concatenate the new row to the existing DataFrame
        df_results = pd.concat([df_results, new_row_df], ignore_index=True)

    metrics_filepath = get_filepath_from_results_path(
        results_path=results_path,
        folder_name="metrics",
        file_name="metrics_results.csv",
        is_cv_loop=is_cv_loop,
        output_path=output_path)
    
    df_results.to_csv(metrics_filepath)

    return metrics_filepath


def main():
    """
    The Main Program.
    """
    args = parse_command_line_args()
    config_dict = get_config(args['file'])
    ensure_path_exists(config_dict['results_path'])
    results_path = Path(config_dict['results_path'])
    
    output_path = config_dict.get('output_path', None)
    if output_path is not None:
        output_path = Path(output_path)

    is_cv_loop = config_dict.get('is_cv_loop', None)
    metrics_list = config_dict.get('metrics_list', None)
    if not isinstance(metrics_list, list):
        metrics_list = [metrics_list]
        
    generate_metrics_file(
        metrics_list=metrics_list,
        results_path=results_path,
        output_path=output_path,
        is_cv_loop=is_cv_loop)
    

if __name__ == "__main__":
    main()
