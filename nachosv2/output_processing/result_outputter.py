from typing import Optional, Callable, Union, Tuple, List
from pathlib import Path
import os
import fasteners
from termcolor import colored

from torch import nn
import pandas as pd
from nachosv2.model_processing.predict_model import predict_model


def save_dict_or_listdict_to_csv(values: Union[dict,List[dict]],
                                 path: Path,
                                 filename: str,
                                 overwrite: bool,
                                 verbose: bool = True):
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
    filepath = path / filename
    
    if filepath.exists() and not overwrite:
        return
    
    values_df.to_csv(filepath)
    if verbose:
        print(f"File {filename} stored at {path}")


def get_prefix_and_folder_path(test_fold: str,
                               hp_config_index: int,
                               validation_fold: Optional[str],
                               is_cv_loop: bool,
                               output_path: Path) -> tuple[str, Path]:
    
    prefix = f'test_{test_fold}_hpconfig_{hp_config_index}'

    # TODO:
    # Add cv_loop or ct_loop

    loop_folder ="CV" if is_cv_loop else "CT"
    path_folder_output = output_path / loop_folder / 'training_results' / \
                         f'test_{test_fold}' / f"hpconfig_{hp_config_index}"
        
    if is_cv_loop:
        val_folder = f"val_{validation_fold}" 
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

    filename_map = {
        "validation": f"{file_prefix}_prediction_val.csv",
        "test": f"{file_prefix}_prediction_test.csv"
    }

    filename = filename_map[partition_type]
    
    filepath = output_path / filename
    
    if filepath.exists():
        return
    
    preds, pred_probs, true_labels, filepaths = predict_model(
                                                execution_device,
                                                trained_model,
                                                partitions_info_dict[partition_type]['dataloader'])
   
    prediction_rows = []
    num_classes = len(pred_probs[0])
    
    for index in range(len(preds)):
        dict_temp = {}
        dict_temp["index"] = index
        dict_temp["filename"] = Path(filepaths[index]).name
        dict_temp["filepath"] = filepaths[index]
        dict_temp["true_label"] = true_labels[index]
        dict_temp["predicted_class"] = preds[index]
        for i in range(num_classes):
            dict_temp[f"class_{i}_prob"] = pred_probs[index][i]
        prediction_rows.append(dict_temp)

    save_dict_or_listdict_to_csv(
        values=prediction_rows,
        path=output_path,
        filename=filename,
        overwrite=True,
        verbose=True,
        )


def save_dict_to_csv(dictionary: dict,
                     output_path: Path,
                     test_fold: str,
                     hp_config_index: int,
                     validation_fold: str,
                     is_cv_loop: bool,
                     suffix: str,
                     verbose: bool = True):
    
    prefix, path_folder_output = get_prefix_and_folder_path(test_fold,
                                                            hp_config_index,
                                                            validation_fold,
                                                            is_cv_loop,
                                                            output_path)

    # Saves the history
    save_dict_or_listdict_to_csv(
        values=dictionary,
        path=path_folder_output,
        filename=f"{prefix}_{suffix}.csv",
        overwrite=True,
        verbose=verbose,
        )


def predict_and_save_results(execution_device: str,
                             output_path: Path,
                             test_fold: str,
                             hp_config_index: int,
                             validation_fold: str,
                             model: nn.Module,
                             time_elapsed: float,
                             partitions_info_dict: dict,
                             class_names: List[str],
                             is_cv_loop: bool,
                             enable_prediction_on_test: bool=None):
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

    
    prefix, path_folder_output = get_prefix_and_folder_path(test_fold,
                                                            hp_config_index,
                                                            validation_fold,
                                                            is_cv_loop,
                                                            output_path)
            
    # Saves Class/Categories indices with corresponding names   
    categories_info =[{"index": index, "class_name": class_names[index]} \
                      for index in range(len(class_names))]
        
    save_dict_or_listdict_to_csv(
        values=categories_info,
        path=path_folder_output,
        filename=f"{prefix}_class_names.csv",
        overwrite=True,
        verbose=True,
        ) 
    
    # Saves Total time of training
    time_info = [{"time_total": time_elapsed}]
    save_dict_or_listdict_to_csv(
        values=time_info,
        path=path_folder_output,
        filename=f"{prefix}_time_total.csv.csv",
        overwrite=True,
        verbose=True,
        )


    # Adds the predictions et true labels to the metric dictionary
    if is_cv_loop: # For the cross-testing loop
        save_prediction_results("validation",
                                execution_device,
                                model,
                                partitions_info_dict,
                                path_folder_output,
                                prefix)
        
        end_message = f"Finished writing results to file " + \
                      f"test fold '{test_fold}' and validation subject " + \
                      f"'{validation_fold}'.\n"
                      
        if enable_prediction_on_test:
            save_prediction_results("test",
                                    execution_device,
                                    model,
                                    partitions_info_dict,
                                    path_folder_output,
                                    prefix)
        
            end_message = f"Finished writing results to file " + \
                        f"test fold '{test_fold}' and validation subject " + \
                        f"'{validation_fold}'.\n"
            
    else: # For the cross-validation loop
        save_prediction_results("test",
                                execution_device,
                                model,
                                partitions_info_dict,
                                path_folder_output,
                                prefix)
        end_message = f"Finished writing results to file " + \
                      f"test fold '{test_fold}'.\n"
    
    print(colored(end_message, 'green'))
