from termcolor import colored

from nachosv2.training.training_processing.training_variables import TrainingVariables
from nachosv2.training.training_processing.training_loop import training_loop


def sequential_subject_loop(execution_device, current_configuration, test_subject_name, data_dictionary, number_of_epochs, normalize_transform, is_outer_loop, is_3d = False, is_verbose_on = False):
    """
    Executes the training loop for the given test subject.

    Args:
        execution_device (str): The name of the device that will be use.
        current_configuration (dict): The training configuration.
        
        test_subject_name (str): The test subject name.
        data_dictionary (dict of list): The data dictionary.
        number_of_epochs (int): The number of epochs.
        
        is_outer_loop (bool): If this is of the outer loop.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
    """
    
    print(colored(
        f"\n========== " + 
        f"Starting training for {test_subject_name} in {current_configuration['selected_model_name']}" +
        f" ==========\n",
        'green'))
    
    
    # Gets the needed data for training
    training_variables = TrainingVariables(current_configuration, test_subject_name, is_outer_loop, is_verbose_on)
    

    # Starts the training loop
    training_loop(
        execution_device,                   # The name of the device that will be use
        current_configuration,              # The input configuration
        test_subject_name,                  # The name of the testing subject
        training_variables.fold_list,       # A list of fold partitions
        training_variables.number_of_folds, # the number of rotations to perform
        data_dictionary,                    # The data dictionary
        number_of_epochs,                   # The number of epochs
        normalize_transform,
        is_outer_loop,                      # If this is of the outer loop
        is_3d,
        is_verbose_on = is_verbose_on       # If the verbose mode is activated
    )
