from pathlib import Path
from typing import List
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
    'f1': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='binary'),
    'f1_micro': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='micro'),
    'f1_macro': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='macro'),
    'f1_weighted': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='weighted'),
    'f1_samples': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='samples'),
    'neg_log_loss': metrics.log_loss,
    'precision': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average='binary'),
    'recall': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average='binary'),
    'jaccard': lambda y_true, y_pred: metrics.jaccard_score(y_true, y_pred, average='binary'),
    'roc_auc': metrics.roc_auc_score,
    'roc_auc_ovr': metrics.roc_auc_score,
    'roc_auc_ovo': metrics.roc_auc_score,
    'roc_auc_ovr_weighted': metrics.roc_auc_score,
    'roc_auc_ovo_weighted': metrics.roc_auc_score,
    # Add clustering metrics here as needed
}


def generate_individual_metric(list_metrics: List[str],
                               path: Path) -> dict:
    
    df_results = pd.read_csv(path)
    actual = df_results['true_label']
    predicted = df_results['predicted_class']
    
    metrics_dict = {}
    # Get the corresponding function and compute the metric
    for metric in list_metrics:
        if metric in metric_functions:
            metric_function = metric_functions[metric]
            score = metric_function(actual, predicted)
            metrics_dict[metric] = [score]        
        else:
            raise ValueError(f"Metric {metric} not supported.")
    
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