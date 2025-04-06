from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict
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


def group_predictions_by_metadata(predictions_path_list,
                                  is_cv_loop) -> Dict[tuple, List[Path]]:
    """
    Groups prediction file paths based on parsed metadata extracted from filenames.

    This is useful for organizing predictions by common attributes such as 
    test fold, hp config, or validation fold.

    Args:
        predictions_path_list (List[Path]): List of file paths for predictions.
        is_cv_loop (bool): Flag indicating if cross-validation is used.

    Returns:
        Dict[Tuple, List[Path]]: A dictionary where keys are tuples of parsed filename info 
                                 and values are lists of corresponding file paths.
    """
    grouped_predictions = defaultdict(list)

    for predictions_path in predictions_path_list:
        # Parse filename to extract metadata (e.g., test fold, hp config, val fold)
        metadata = parse_filename(predictions_path, is_cv_loop)

        # Use metadata tuple as a key to group files
        key = tuple(metadata.values())
        grouped_predictions[key].append(predictions_path)
        
    return grouped_predictions


def create_dictionary_from_tuple(tuple_key: tuple) -> dict:
    """
    Creates a dictionary from a tuple.

    Args:
        tuple_key (tuple): The tuple to be converted to a dictionary.

    Returns:
        dict: The dictionary created from the tuple.
    """
    dict_return = {}
    if len(tuple_key) == 3:
        dict_return["test_fold"]=tuple_key[0]
        dict_return["hp_config"]=tuple_key[1]
        dict_return["val_fold"]=tuple_key[2]
    elif len(tuple_key) == 2:
        dict_return["test_fold"]=tuple_key[0]
        dict_return["hp_config"]=tuple_key[1]

    return dict_return    


def generate_metrics_file(metrics_list: List[str],
                          results_path: Path,
                          is_cv_loop: bool,
                          output_path: Optional[Path]) -> Path:
    """
    Generates a CSV file containing evaluation metrics for machine learning predictions.

    Args:
        metrics_list (List[str]): A list of metric names to be extracted from each prediction results file.
        results_path (Path): Path to the directory containing prediction results.
        is_cv_loop (bool): Specifies whether cross-validation (CV) is being used. If True, results 
            include cross-validation results.
        output_path (Optional[Path]): Directory where the generated metrics file should be saved. 
            If None, the default results path will be used.

    Returns:
        Path: The file path of the generated metrics CSV.
    """
        
    # Define filename suffix for prediction result files 
    suffix_filename = "prediction"
    
    # Get the list of file paths containing prediction results
    predictions_path_list = get_filepath_list(
        directory_to_search_path=results_path,
        string_in_filename=suffix_filename,
        is_cv_loop=is_cv_loop)
    
    # TODO: Improve name of variables and comments 
    metadata_predictions_dict = group_predictions_by_metadata(
                                    predictions_path_list,
                                    is_cv_loop)
    
    # Initialize an empty DataFrame to store metrics results
    df_results = pd.DataFrame()

    # Process each prediction file
    for metadata_key, associated_prediction_files in metadata_predictions_dict.items():
        # Generate a dictionary of extracted metrics
        metrics_dict = generate_individual_metric(metrics_list,
                                                  associated_prediction_files)
        summary_dict = create_dictionary_from_tuple(metadata_key)
        summary_dict.update(metrics_dict)
        new_row_df = pd.DataFrame(summary_dict)

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
