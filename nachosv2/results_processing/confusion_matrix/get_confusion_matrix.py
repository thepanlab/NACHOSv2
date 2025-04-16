from pathlib import Path
from typing import Optional
from termcolor import colored
from sklearn import metrics
import pandas as pd
from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.get_config import get_config
from nachosv2.setup.utils import get_other_result
from nachosv2.setup.utils import get_filepath_list
from nachosv2.setup.utils import get_new_filepath_from_suffix
from nachosv2.setup.files_check import ensure_path_exists


def generate_individual_confusion_matrix(path: Path) -> pd.DataFrame:
    """
    Generate a confusion matrix DataFrame from prediction results CSV file.

    This function reads a CSV file located at the provided `path`, which is expected to contain at least 
    two columns: "true_label" and "predicted_class". It computes the confusion matrix using 
    scikit-learn's `metrics.confusion_matrix` function. Additionally, it retrieves an associated class 
    names CSV file (using the `get_other_result` function) that provides the mapping between class indices 
    and class names. The retrieved class names are used to create a multi-index for both the rows (labeled 
    as "Ground truth") and the columns (labeled as "Predicted") in the output DataFrame.

    Args:
        path (Path): prediction CSV filepath

    Returns:
        pd.DataFrame: A DataFrame containing the confusion matrix with a hierarchical index. The row 
        index is prefixed with "Ground truth" and the column index with "Predicted", with the second level 
        of the indices corresponding to the class names.

    Raises:
        FileNotFoundError: If the CSV file at the provided path or the associated class names file is not found.
    """
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
                is_cv_loop: bool):
    """
    Generate and save confusion matrix CSV files from prediction result files.

    This function searches for prediction result files within the provided directory 
    (`results_path`) using a specific filename suffix ("prediction"). For each prediction 
    file found, it generates an individual confusion matrix by calling 
    `generate_individual_confusion_matrix()`. The resulting confusion matrix is then saved 
    as a CSV file to a new path computed by `get_newfilepath_from_predictions()`. 
    If the `is_cv_loop` flag is set, predictions are retrieved from the `CV` folder; otherwise,
    they are sourced from the `CT` folder.
    Args:
        results_path (Path): The directory containing prediction result CSV files.
        output_path (Optional[Path]): The directory where the confusion matrix CSV files 
                                      will be saved.
        is_cv_loop (Optional[bool], optional): Flag indicating whether the prediction results 
                                               are from a cross-validation loop, which affects 
                                               file naming conventions. Defaults to None.

    Returns:
        None
    """
    suffix_filename = "prediction"
    prediction_file_path_list = get_filepath_list(results_path,
                                                  suffix_filename,
                                                  is_cv_loop)

    # Extract and print test and validation fold numbers
    for predictions_path in prediction_file_path_list:        
        cf_df = generate_individual_confusion_matrix(predictions_path)

        cf_filepath = get_new_filepath_from_suffix(predictions_path,
                                                   "prediction",
                                                   "confusion_matrix",
                                                   is_cv_loop,
                                                   output_path)

        cf_df.to_csv(cf_filepath)


def main():
    """
    The Main Program.
    """
    args = parse_command_line_args()
    config_dict = get_config(args['file'])
    results_path = Path(config_dict['results_path'])
    ensure_path_exists(results_path)
    
    output_path = config_dict.get('output_path', None)
    if output_path is not None:
        output_path = Path(output_path)

    is_cv_loop = config_dict.get('is_cv_loop')
    
    generate_cf(results_path, output_path, is_cv_loop)


if __name__ == "__main__":
    main()
