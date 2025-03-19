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
    'precision': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average='binary'),
    'recall': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average='binary'),
    'jaccard': lambda y_true, y_pred: metrics.jaccard_score(y_true, y_pred, average='binary'),
    # ROC AUC Scores - Specifying Multiclass/Multi-label Cases
    'roc_auc': lambda y_true, y_pred: metrics.roc_auc_score(y_true, y_pred),
    'roc_auc_ovr': lambda y_true, y_pred: metrics.roc_auc_score(y_true, y_pred, multi_class='ovr'),
    'roc_auc_ovo': lambda y_true, y_pred: metrics.roc_auc_score(y_true, y_pred, multi_class='ovo'),
    'roc_auc_ovr_weighted': lambda y_true, y_pred: metrics.roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted'),
    'roc_auc_ovo_weighted': lambda y_true, y_pred: metrics.roc_auc_score(y_true, y_pred, multi_class='ovo', average='weighted')
}


def generate_individual_metric(metrics_list: List[str],
                               path: Path) -> Dict[str, float]:
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

    df_results = pd.read_csv(path)
    actual = df_results['true_label']
    predicted = df_results['predicted_class']
    
    metrics_dict = {}
    # Get the corresponding function and compute the metric
    for metric in metrics_list:
        if metric in metric_functions:
            metrics_dict[metric] = metric_functions[metric](actual,predicted)       
        else:
            message_str = f"Metric {metric} not supported. Metrics supported: "
            
            for key in metric_functions:
                message_str += f"{key}, "
                
            raise ValueError(message_str)
    
    return metrics_dict


def parse_filename(filepath: Path,
                   is_cv_loop: bool) -> dict:
    """
    Parse parts from the filename to extract test fold, hyperparameter configuration, and validation fold.
    """
    parts = filepath.stem.split('_')
    return {
        'test_fold': [parts[1]],  # Assumes specific filename structure
        'hp_config': [int(parts[3])],
        'val_fold': [parts[5]] if is_cv_loop else None
    }
    

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