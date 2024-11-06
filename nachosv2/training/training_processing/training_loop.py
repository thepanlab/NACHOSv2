from termcolor import colored

from nachosv2.training.training_processing.training_fold import TrainingFold
from nachosv2.log_processing.read_log import read_item_list_in_log
from nachosv2.log_processing.write_log import write_log


def training_loop(execution_device, configuration, testing_subject, fold_list, number_of_rotations, data_dictionary, number_of_epochs, normalize_transform, is_outer_loop, is_3d = False, rank = None, is_verbose_on = False):
    """
    Creates a model, trains it, and writes its outputs.
        
    Args:
        execution_device (str): The name of the device that will be use.
        configuration (dict): The input configuration.
        
        testing_subject (str): The name of the testing subject.
        fold_list (list of dict): A list of fold partitions.
        number_of_rotations (int): the number of rotations to perform.
        
        data_dictionary (dict of list): The data dictionary.
        number_of_epochs (int): The number of epochs.
        
        is_outer_loop (bool): If this is of the outer loop. Default is false. (Optional)
        rank (int): The process rank. Default is none. (Optional)
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
    """
    
    print(colored(f'Beginning the training loop for {testing_subject}.', 'green'))
    
    
    # Gets the current rotation progress
    current_rotation = 0
    log_rotations = read_item_list_in_log(
        configuration['output_path'],
        configuration['job_name'], 
        ['current_rotation'],
        rank
    )
    
    
    # If a current_rotation exists in the log, sets the next current_rotation to that value
    if log_rotations and 'current_rotation' in log_rotations and testing_subject in log_rotations['current_rotation']:
        current_rotation = log_rotations['current_rotation'][testing_subject]
        print(colored(f'Starting off from rotation {current_rotation + 1} for testing subject {testing_subject}.', 'cyan'))
    
    
    # Trains for every current_rotation specified
    for rotation in range(current_rotation, number_of_rotations):
        
        # Outer loop
        if is_outer_loop:
            validation_subject = fold_list[rotation]['training'][0]
            print(colored(f'--- Rotation {rotation + 1}/{number_of_rotations} for test subject {testing_subject} ---', 'magenta'))
        
        # Inner loop will use the validation subjects
        else:
            validation_subject = fold_list[rotation]['validation'][0]
            print(colored(f'--- Rotation {rotation + 1}/{number_of_rotations} for test subject {testing_subject} and val subject {validation_subject} ---', 'magenta'))
        
        
        # Creates and runs the training fold for this subject pair
        training_fold = TrainingFold(
            execution_device,       # The name of the device that will be use
            rotation,               # The fold index within the loop
            configuration,          # The training configuration
            testing_subject,        # The test subject name
            validation_subject,     # The validation_subject name
            fold_list,              # A list of fold partitions
            data_dictionary,        # The data dictionary
            number_of_epochs,       # The number of epochs
            normalize_transform,    # 
            rank,                   # An optional value of some MPI rank. Default is none. (Optional)
            is_outer_loop,          # If this is of the outer loop. Default is false. (Optional)
            is_3d,                  # 
            is_verbose_on           # If the verbose mode is activated. Default is false. (Optional)
        )
        
        training_fold.run_all_steps()
        
        
        # Writes the index to log
        write_log(
            config = configuration,
            testing_subject = testing_subject,
            rotation = rotation,
            log_directory = configuration['output_path'],
            log_rotations = None,
            validation_subject = None,
            rank = None,
            is_outer_loop = False
        )
        