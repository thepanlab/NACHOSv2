from pathlib import Path
import re
from termcolor import colored
from typing import Optional, List
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


def generate_metrics_file(list_metrics: List[str],
                          results_path: Path,
                          output_path: Optional[Path],
                          is_cv_loop: Optional[bool]):
    
    suffix_filename = "prediction_results"
    predictions_path_list = get_filepath_list(
        directory_to_search_path=results_path,
        string_in_filename=suffix_filename,
        is_cv_loop=is_cv_loop)
    
    df_results = pd.DataFrame()
    # Extract and print test and validation fold numbers
    for predictions_path in predictions_path_list:          
        metrics_dict = generate_individual_metric(list_metrics,
                                                  predictions_path)
    
        file_info = parse_filename(predictions_path, is_cv_loop)

        if not is_cv_loop:
            del file_info['val_fold']
        
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
    list_metrics = config_dict.get('list_metrics', None)
    if not isinstance(list_metrics, list):
        list_metrics = [list_metrics]
        
    generate_metrics_file(
        list_metrics=list_metrics,
        results_path=results_path,
        output_path=output_path,
        is_cv_loop=is_cv_loop)
    

if __name__ == "__main__":
    main()
