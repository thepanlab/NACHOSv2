from pathlib import Path
import re
from termcolor import colored
from typing import Optional
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


def generate_individual_confusion_matrix(path: Path) -> pd.DataFrame:
    
    df_results = pd.read_csv(path)
    actual = df_results['true_label']
    predicted = df_results['predicted_class']
    
    confusion_matrix = metrics.confusion_matrix(actual, predicted) 

    # get class_names file to use in confusion matrices
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


def generate_cf(results_path: Path,
                output_path: Optional[Path],
                is_cv_loop: Optional[bool] = None):
    
    suffix_filename = "prediction_results"
    predictions_file_path_list = get_filepath_list(results_path,
                                                   suffix_filename,
                                                   is_cv_loop)
    
    if is_cv_loop is None:
        is_cv_loop = determine_if_cv_loop(predictions_file_path_list[0])
        
    # Extract and print test and validation fold numbers
    for predictions_path in predictions_file_path_list:
        filename = predictions_path.name            
        cf_df = generate_individual_confusion_matrix(predictions_path)

        cf_filepath = get_newfilepath_from_predictions(predictions_path,
                                                       "confusion_matrix",
                                                       output_path)
        
        cf_df.to_csv(cf_filepath)


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
    
    generate_cf(results_path, output_path, is_cv_loop)


if __name__ == "__main__":
    main()
