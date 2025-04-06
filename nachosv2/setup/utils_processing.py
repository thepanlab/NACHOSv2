from pathlib import Path
from typing import List, Dict
from sklearn import metrics
import yaml
from termcolor import colored
import pandas as pd

"""
Green: indications about where is the training
Cyan: verbose mode
Magenta: results
Yellow: warnings
Red: errors, fatal or not
"""

metric_functions = {
    'accuracy': metrics.accuracy_score,
    'balanced_accuracy': metrics.balanced_accuracy_score,
    'top_k_accuracy': metrics.top_k_accuracy_score,
    'average_precision': metrics.average_precision_score,
    'neg_brier_score': lambda y_true, y_pred: -metrics.brier_score_loss(y_true, y_pred),
    # F1 Scores with different averaging methods
    'f1': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='binary'),
    'f1_micro': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='micro'),
    'f1_macro': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='macro'),
    'f1_weighted': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='weighted'),
    'f1_samples': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='samples'),
    'neg_log_loss': metrics.log_loss,
    # Precision, Recall, and Jaccard Score
    'precision_binary': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average='binary'),
    'precision_micro': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average='micro'),
    'precision_macro': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average='macro'),
    'recall_binary': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average='binary'),
    'recall_micro': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average='micro'),
    'recall_macro': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average='macro'),
    'jaccard_binary': lambda y_true, y_pred: metrics.jaccard_score(y_true, y_pred, average='binary'),
    'jaccard_micro': lambda y_true, y_pred: metrics.jaccard_score(y_true, y_pred, average='micro'),
    'jaccard_macro': lambda y_true, y_pred: metrics.jaccard_score(y_true, y_pred, average='macro'),
    # ROC AUC Scores - Specifying Multiclass/Multi-label Cases
    'roc_auc': lambda y_true, y_pred: metrics.roc_auc_score(y_true, y_pred),
    'roc_auc_ovr': lambda y_true, y_pred: metrics.roc_auc_score(y_true, y_pred, multi_class='ovr'),
    'roc_auc_ovo': lambda y_true, y_pred: metrics.roc_auc_score(y_true, y_pred, multi_class='ovo'),
    'roc_auc_ovr_weighted': lambda y_true, y_pred: metrics.roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted'),
    'roc_auc_ovo_weighted': lambda y_true, y_pred: metrics.roc_auc_score(y_true, y_pred, multi_class='ovo', average='weighted')
}


def is_metric_allowed(metric: str) -> bool:
    """
    Check if the metric is allowed.

    Args:
        metric (str): The metric to check.

    Returns:
        bool: True if the metric is allowed, False otherwise.
    """
    return metric in metric_functions   


def get_partition_from_prediction_file(filepath: Path) -> str:
    partition = filepath.stem.split('_')[-1]

    if partition not in [ 'val', 'test']:
        raise ValueError(f"Partition {partition} not recognized. Expected 'val', or 'test'. in {filepath}")

    if partition == 'val':
        partition = 'validation' # for consistency with the results of history


    return partition


def generate_individual_metric(metrics_list: List[str],
                               path_list: List[Path]) -> Dict[str, float]:
    """
    Computes specified evaluation metrics for classification predictions stored in a CSV file.

    Parameters:
    ----------
    metrics_list : List[str]
        A list of metric names to compute. Each metric should correspond to a function 
        defined in `metric_functions`.
    path : Path
        Path to the CSV file containing prediction results with `true_label` and 
        `predicted_class` columns.

    Returns:
    -------
    Dict[str, float]
        A dictionary where keys are metric names and values are the computed metric scores.

    Raises:
    ------
    ValueError
        If any metric in `list_metrics` is not found in `metric_functions`.
    """
    metrics_dict = {}
    # Get the corresponding function and compute the metric
    for metric in metrics_list:
        if metric not in metric_functions:
            message_str = f"Metric {metric} not supported. Metrics supported: "
            for key in metric_functions:
                message_str += f"{key}, "
            raise ValueError(message_str)
        else:
            # in case there are validation and test
            # sort to have validation first
            path_list_sorted = sorted(path_list, reverse=True)
            for path in path_list_sorted:
                partition = get_partition_from_prediction_file(path)
                df_results = pd.read_csv(path)
                actual = df_results['true_label']
                predicted = df_results['predicted_class']
                metrics_dict[f"{partition}_{metric}"] = [metric_functions[metric](actual, predicted)]

    return metrics_dict


def parse_filename(filepath: Path,
                   is_cv_loop: bool) -> dict:
    """
    Parses a filename to extract test fold, hyperparameter configuration, and optionally validation fold.
    
    Args:
        filepath (Path): The file path object containing the filename.
        is_cv_loop (bool): Flag indicating whether the filename contains a validation fold.

    Returns:
        dict: Parsed values with keys 'test_fold', 'hp_config', and optionally 'val_fold'.
    """
    parts = filepath.stem.split('_')

    parsed_data = {
        'test_fold': parts[1],  # Assumes test fold is at index 1
        'hp_config': int(parts[3])  # Assumes hyperparameter config is at index 3
    }

    if is_cv_loop:
        parsed_data['val_fold'] = parts[5]  # Assumes validation fold is at index 5

    return parsed_data


def save_dict_to_yaml(data_dict: dict,
                      file_path: Path):
    """
    Save a dictionary to a YAML file.
    
    Args:
        data_dict (dict): The dictionary to save.
        file_path (Path): The file path where the YAML file will be saved.
        
    Returns:
        None
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data_dict, file, sort_keys=False)
        print(f"Dictionary saved successfully to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")