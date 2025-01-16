import json
import csv
import os
import pandas as pd
from termcolor import colored
from pathlib import Path
from typing import Dict, List, Optional

from torch import nn

from nachosv2.model_processing.predict_model import predict_model


def create_folders(path: Path,
                   names: Optional[List[str]] = None):
    """
    Creates folder(s) if they do not exist.

    Args:
        path (Path): Path to the folder.
        names (list): The folder name(s). (Optional)
    """
    
    if names is not None:
        for name in names:
            folder_path = path / name
            if not folder_path.exists():
                folder_path.mkdir(mode=0o777, parents=True, exist_ok=True)
    else:
        if not path.exists():
            path.mkdir(mode=0o777, parents=True, exist_ok=True)


def save_history(history: Dict[str, List[float]],
                 path_prefix: Path,
                 file_prefix: str):
    if history is not None:
        history_df = pd.DataFrame(history)
        history_df.to_csv(path_prefix / f"{file_prefix}_history.csv")
    
    else:
        print(colored(f'Warning: The model history is empty for: {file_prefix}', 'yellow'))


def metric_writer2(values: List[Dict],
                   path: Path,
                   filename: str):
    """
    Writes some list in a file.

    Args:
        path (str): A file path.
        values (list): A list of values.
        path_prefix (str): The prefix of the file name and directory.
    """
    values_df = pd.DataFrame(values)   
    values_df.to_csv(path / filename)


def metric_writer(path, values, path_prefix):
    """
    Writes some list in a file.

    Args:
        path (str): A file path.
        values (list): A list of values.
        path_prefix (str): The prefix of the file name and directory.
    """
    
    with open(f"{path_prefix}/{path}", 'w', encoding = 'utf-8') as file_pointer:
        
        # Predicted values
        if path.endswith('predicted.csv'):
            writer = csv.writer(file_pointer)
            
            for item in values:
                writer.writerow(item)
                
        # Class names
        elif path.endswith('_class_names.json'): 
            json.dump(values, file_pointer)
            
            
        # Everything else
        else:
            writer = csv.writer(file_pointer)
            
            for item in values:
                writer.writerow([item])





def generate_inner_loop_results(execution_device:str,
                                trained_model: nn.Module,
                                datasets: dict,
                                metrics: dict,
                                file_prefix: str):
    """
    Saves the results of the inner loop to the metrics dictionary.

    Args:
        execution_device (str): The device on which the model is executed.
        trained_model (nn.Module): The trained model to be used for predictions.
        datasets (dict): Dictionary containing 'testing' and 'validation' datasets.
        metrics (dict): Dictionary to store the metrics and results.
        file_prefix (str): Prefix for the file names used in saving the results.
    """
    
    # Predicts probability results for the testing and validation datasets
    # predicted_labels_in_testing, predicted_probabilities_in_testing, true_labels_list_in_testing, file_names_list_in_testing = predict_model(execution_device, trained_model, datasets['testing']['ds'])
    predicted_labels_in_validation, predicted_probabilities_in_validation, true_labels_list_in_validation, file_names_list_in_validation = predict_model(execution_device, trained_model, datasets['validation']['ds'])
    
    
    # Saves predicted probabilities for the testing and validation datasets
    metrics[f"prediction/{file_prefix}_test_predicted_labels.csv"] = [l for l in predicted_labels_in_testing]
    metrics[f"prediction/{file_prefix}_test_predicted_probabilities.csv"] = [l for l in predicted_probabilities_in_testing]
    metrics[f"prediction/{file_prefix}_val_predicted_labels.csv"] = [l for l in predicted_labels_in_validation]
    metrics[f"prediction/{file_prefix}_val_predicted_probabilities.csv"] = [l for l in predicted_probabilities_in_validation]
    
    # Saves true labels for the testing and validation datasets
    metrics[f"true_label/{file_prefix}_test_true_label.csv"] = true_labels_list_in_testing
    metrics[f"true_label/{file_prefix}_val_true_label.csv"] = true_labels_list_in_validation
    
    # Saves file names for the testing and validation datasets
    metrics[f'file_name/{file_prefix}_test_file.csv'] = file_names_list_in_testing
    metrics[f'file_name/{file_prefix}_val_file.csv'] =  file_names_list_in_validation
