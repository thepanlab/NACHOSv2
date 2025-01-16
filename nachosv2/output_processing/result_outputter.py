import fasteners
import os
from termcolor import colored
from pathlib import Path
from typing import Optional, Callable, Union, Tuple, List

from torch import nn
import pandas as pd
# from nachosv2.output_processing.result_outputter_utils import create_folders, save_history, save_outer_loop, save_inner_loop, metric_writer, metric_writer2
from nachosv2.model_processing.evaluate_model import evaluate_model
from nachosv2.model_processing.save_model import save_model
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

def metric_writer(values: List[dict],
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


def save_outer_loop(execution_device, trained_model,
                    partitions_info_dict, metrics,
                    file_prefix):
    """
    Saves the results of the outer loop to the metrics dictionary.

    Args:
        execution_device (str): The device on which the model is executed.
        trained_model (nn.Module): The trained model to be used for predictions.
        partitions_info_dict (dict): Dictionary containing 'testing' datasets.
        metrics (dict): Dictionary to store the metrics and results.
        file_prefix (str): Prefix for the file names used in saving the results.
    """
    
    # Predicts probability results for the testing dataset
    # predicted_labels_in_testing, predicted_probabilities_in_testing, true_labels_list_in_testing, file_names_list = predict_model(execution_device, trained_model, partitions_info_dict['test']['ds'])
    
    prediction_list, prediction_probabilities_list, true_labels_list = \
        predict_model(execution_device, trained_model, partitions_info_dict['test']['ds'])
    
    
    # Saves predicted probabilities for the testing dataset
    metrics[f"prediction/{file_prefix}_predicted_labels.csv"] = [l for l in predicted_labels_in_testing]
    metrics[f"prediction/{file_prefix}_predicted_probabilities.csv"] = [l for l in predicted_probabilities_in_testing]
    
    # Saves true labels for the testing dataset
    metrics[f"true_label/{file_prefix}_true_label.csv"] = true_labels_list_in_testing
    
    # Saves file names for the testing dataset
    metrics[f'file_name/{file_prefix}_file.csv'] = file_names_list
       
    
def save_inner_loop(execution_device: str,
                    trained_model: nn.Module,
                    partitions_info_dict: dict,
                    output_path: Path,
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
    preds, pred_probs, true_labels = predict_model(
                                        execution_device,
                                        trained_model,
                                        partitions_info_dict['validation']['dataloader'])
   
    prediction_rows = []
    num_classes = len(pred_probs[0])
    
    for index in range(len(preds)):
        dict_temp = {}
        dict_temp["index"] = index
        dict_temp["predicted_class"] = preds[index]
        for i in range(num_classes):
            dict_temp[f"class_{i}_prob"] = pred_probs[index][i]
        prediction_rows.append(dict_temp)

    metric_writer(prediction_rows,
                   output_path,
                   f"{file_prefix}_prediction_results.csv")    


def output_results(execution_device: str,
                   output_path: Path,
                   test_fold_name: str,
                   validation_fold_name: str,
                   model: nn.Module,
                   history: dict,
                   time_elapsed: float,
                   partitions_info_dict: dict,
                   class_names: List[str],
                   job_name: str,
                   architecture_name: str,
                   loss_function: nn.CrossEntropyLoss,
                   is_outer_loop: bool,
                   rank: int):
    """
    Outputs results from the trained model.
        
    Args:
        execution_device (str): The execution device.
        output_path (str): Where to output the results.
        
        testing_subject (str): The testing subject name.
        validation_subject (str): The validation subject name.
        
        trained_model (TrainingModel): The trained model.
        history (dict): The history outputted by the fitting function.
        
        time_elapsed (double): The elapsed time from the fitting phase.
        datasets (dict): A dictionary of various values for the data-splits.
        class_names (list of str): The class names of the data.
        
        job_name (str): The name of this config's job name.
        config_name (str): The name of this config's config (model) name.
        
        loss_function (nn.CrossEntropyLoss): The loss function.
        
        is_outer_loop (bool): If this is of the outer loop.
        rank (int): The process rank. May be None.
    """

    # Creates the file prefix
    if is_outer_loop:
        file_prefix = f"{architecture_name}_test_{test_fold_name}"
    
    else:
        file_prefix = f"{architecture_name}_test_{test_fold_name}_val_{validation_fold_name}"
    
    # Creates the path prefix
    path_folder_output = output_path / f'Test_subject_{test_fold_name}' / \
                  f'config_{architecture_name}' / file_prefix
    
    # Saves the model
    # save_model(trained_model, f"{path_prefix}/model/{file_prefix}_{trained_model.model_type}.pth")
    
    # Saves the history
    # save_history(history, path_folder_output, file_prefix)
    metric_writer(history,
                  path_folder_output,
                  f"{file_prefix}_history.csv")
    # Writes the class names   
    categories_info =[{"index": indexes, "class_name": class_names[indexes]} \
                       for indexes in range(len(class_names))]
   
    metric_writer(categories_info,
                  path_folder_output,
                  f"{file_prefix}_class_names.csv") 
    
    # Creates the metrics dictionary and adds the training time
    time_info = [{"time_total": time_elapsed}]

    metric_writer(time_info,
                  path_folder_output,
                  f"{file_prefix}_time_total.csv.csv")

    # TODO: add filename to save_inner_loop
    # Change names and functions in this script: result_outputter.py
    # modify save_outer_loop

    # Adds the predictions et true labels to the metric dictionary
    if is_outer_loop: # For the outer loop
        save_outer_loop(execution_device, model,
                        partitions_info_dict, metric,
                        file_prefix)
    else: # For the inner loop
        save_inner_loop(execution_device,
                        model,
                        partitions_info_dict,
                        path_folder_output,
                        file_prefix)
    
    print(colored(f"Finished writing results to file for {architecture_name}'s "
                  f"test fold name {test_fold_name} and validation subject "
                  f"{validation_fold_name}.\n", 'green'))
