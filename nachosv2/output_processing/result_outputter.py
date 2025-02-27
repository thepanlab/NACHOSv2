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


def save_csv_from_list_dict(values: List[dict],
                           path: Path,
                           filename: str):
    """
    Writes some list in a file.

    Args:
        path (str): A file path.
        values (list): A list of values.
        path_prefix (str): The prefix of the file name and directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    # create directory for path
    values_df = pd.DataFrame(values)   
    values_df.to_csv(path / filename)

def get_prefix_and_folder_path(test_fold_name: str,
                               validation_fold_name: Optional[str],
                               is_cv_loop: bool,
                               output_path: Path) -> tuple[str, Path]:
    
    prefix = f'test_{test_fold_name}'

    path_folder_output = output_path / 'training_results' / \
                        f'test_{test_fold_name}'      
    
    if is_cv_loop:
        val_folder = f"val_{validation_fold_name}" 
        path_folder_output = path_folder_output / val_folder
        
        prefix += f"_{val_folder}"
        
    return prefix, path_folder_output


def save_prediction_results(partition_type: str,
                            execution_device: str,
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
    
    if partition_type not in ["validation", "test"]:
        raise ValueError(f"partition_type should be either 'validation' or 'test', but got {partition_type}.")
    
    # Predicts probability results for the testing and validation datasets
    # TODO: stored filepath
    # verify full filepath and filename !!!
    preds, pred_probs, true_labels, filepaths = predict_model(
                                                execution_device,
                                                trained_model,
                                                partitions_info_dict[partition_type]['dataloader'])
   
    prediction_rows = []
    num_classes = len(pred_probs[0])
    
    for index in range(len(preds)):
        dict_temp = {}
        dict_temp["index"] = index
        dict_temp["filepath"] = filepaths[index]
        dict_temp["true_label"] = true_labels[index]
        dict_temp["predicted_class"] = preds[index]
        for i in range(num_classes):
            dict_temp[f"class_{i}_prob"] = pred_probs[index][i]
        prediction_rows.append(dict_temp)

    save_csv_from_list_dict(prediction_rows,
                            output_path,
                            f"{file_prefix}_prediction_results.csv")    


def save_history_to_csv(history: dict,
                        output_path: Path,
                        test_fold_name: str,
                        validation_fold_name: str,
                        is_cv_loop: bool,
                        rank: int=None):
    
    prefix, path_folder_output = get_prefix_and_folder_path(test_fold_name,
                                                            validation_fold_name,
                                                            is_cv_loop,
                                                            output_path)
        
    # Saves the history
    save_csv_from_list_dict(history,
                            path_folder_output,
                            f"{prefix}_history.csv")


def predict_and_save_results(execution_device: str,
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
                             is_cv_loop: bool):
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

    
    prefix, path_folder_output = get_prefix_and_folder_path(test_fold_name,
                                                            validation_fold_name,
                                                            is_cv_loop,
                                                            output_path)
            
    # Saves Class/Categories indices with corresponding names   
    categories_info =[{"index": index, "class_name": class_names[index]} \
                      for index in range(len(class_names))]
    save_csv_from_list_dict(categories_info,
                            path_folder_output,
                            f"{prefix}_class_names.csv") 
    
    # Saves Total time of training
    time_info = [{"time_total": time_elapsed}]
    save_csv_from_list_dict(time_info,
                            path_folder_output,
                            f"{prefix}_time_total.csv.csv")


    # Adds the predictions et true labels to the metric dictionary
    if is_cv_loop: # For the cross-testing loop
        save_prediction_results("validation",
                                execution_device,
                                model,
                                partitions_info_dict,
                                path_folder_output,
                                prefix)
    else: # For the cross-validation loop
        save_prediction_results("test",
                                execution_device,
                                model,
                                partitions_info_dict,
                                path_folder_output,
                                prefix)
    
    print(colored(f"Finished writing results to file "
                  f"test fold '{test_fold_name}' and validation subject "
                  f"'{validation_fold_name}'.\n", 'green'))
