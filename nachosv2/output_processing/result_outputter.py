import fasteners
import os
from termcolor import colored

from nachosv2.output_processing.result_outputter_utils import create_folders, save_history, save_outer_loop, save_inner_loop, metric_writer
from nachosv2.model_processing.evaluate_model import evaluate_model
from nachosv2.model_processing.save_model import save_model


def output_results(execution_device, output_path, testing_subject, validation_subject, trained_model, history, time_elapsed, datasets, class_names, job_name, config_name, loss_function, is_outer_loop, rank):
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
        file_prefix = f"{trained_model.model_type}_test_{testing_subject}"
    
    else:
        file_prefix = f"{trained_model.model_type}_test_{testing_subject}_val_{validation_subject}"
    
    # Creates the path prefix
    path_prefix = os.path.join(output_path, f'Test_subject_{testing_subject}', f'config_{job_name}_{config_name}', file_prefix)
    
    
    # Creates the folders to output into
    if rank is not None: # If MPI, lock this section else the processes may error out
        with fasteners.InterProcessLock(os.path.join(os.path.dirname(output_path), 'output_lock.tmp')):
            create_folders(path_prefix, ['prediction', 'true_label', 'file_name', 'model'])
      
    else:
        create_folders(path_prefix, ['prediction', 'true_label', 'file_name', 'model'])
    
    
    # Saves the model
    save_model(trained_model, f"{path_prefix}/model/{file_prefix}_{trained_model.model_type}.pth")
    
    
    # Saves the history
    save_history(history, path_prefix, file_prefix)
    
    
    # Writes the class names
    class_output = {class_names[indexes]: indexes for indexes in range(len(class_names))}
    metric_writer(f"{file_prefix}_class_names.json", class_output, path_prefix)
    
    
    # Creates the metrics dictionary and adds the training time
    metrics = {f"{file_prefix}_time_total.csv": [time_elapsed]}


    # Adds the predictions et true labels to the metric dictionary
    if is_outer_loop: # For the outer loop
        save_outer_loop(execution_device, trained_model, datasets, metrics, file_prefix)

    else: # For the inner loop
        save_inner_loop(execution_device, trained_model, datasets, metrics, file_prefix)


    # If possible, writes the metrics using the testing dataset
    if datasets['testing']['ds'] is None:
        print(colored(
            f"Non-fatal Error: evaluation was skipped for the test subject {testing_subject} and validation subject {validation_subject}. " + 
            f"There were no files in the testing dataset.",
            'yellow'
        ))
    
    else:
        metrics[f"{file_prefix}_test_evaluation.csv"] = evaluate_model(execution_device, trained_model, datasets['testing']['ds'], loss_function) # The evaluation results           


    # Writes all metrics to file
    for metric in metrics:
        metric_writer(metric, metrics[metric], path_prefix)
    
    
    print(colored(f"Finished writing results to file for {trained_model.model_type}'s testing subject {testing_subject} and validation subject {validation_subject}.\n", 'green'))
