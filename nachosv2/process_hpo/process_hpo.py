import os
import sys
from pathlib import Path

import re
from termcolor import colored
from sklearn import metrics
import pandas as pd

from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.get_filepath_list import get_filepath_list
from nachosv2.setup.get_config_list import get_config_list

"""
Green: indications about where is the training
Cyan: verbose mode
Magenta: results
Yellow: warnings
Red: errors, fatal or not
"""


def generate_individual_metric(config: dict,
                               path: Path,
                               metric: str) -> pd.DataFrame:
    
    df_results = pd.read_csv(path)
    actual = df_results['true_label']
    predicted = df_results['predicted_class']
    
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
    
    # Get the corresponding function and compute the metric
    if metric in metric_functions:
        metric_function = metric_functions[metric]
        score = metric_function(actual, predicted)
        # return pd.DataFrame([[metric, score]], columns=['Metric', 'Score'])
    else:
        raise ValueError(f"Metric {metric} not supported.")
    
    return score


def main():
    # Parses the command line arguments
    args = parse_command_line_args()
    
    # Defines the arguments
    config_dict_list = get_config_list(args['file'], args['folder'])
    
    is_verbose_on = args['verbose']
    metric = args['metric']
    
    # create summary 
    for config in config_dict_list:
        if not Path(config['data_path']).exists():
            raise ValueError(print(colored(f"Path {config['data_path']} does not exist.", "red")))
        folder_training_results_path = Path(config['data_path'])
        
        string_filename = "prediction_results"
        predictions_file_path_list = get_filepath_list(folder_training_results_path,
                                                       string_filename)
        
        # TODO generate csv all summary
        # TODO generate csv for best 
        
        # Regex to match test and validation info
        pattern = r"test_([A-Za-z0-9]+)_val_([A-Za-z0-9]+)"

        # Extract and print test and validation fold numbers
        for path in predictions_file_path_list:
            filename = path.name
            match = re.search(pattern, filename)
            if match:
                test_fold, val_fold = match.groups()
                print(f"File: {filename}")
                print(f"  Test fold: {test_fold}")
                print(f"  Validation fold: {val_fold}")
                print("-----")

            value_metric = generate_individual_metric(config, path,
                                                      metric)

            # To complete
            
            cf_filepath = get_filepath_confusion_matrix(config,
                                                        path)
            cf_df.to_csv(cf_filepath)
        
        print("Hola")
        

# Sequential Inner Loop
if __name__ == "__main__":
    main()
