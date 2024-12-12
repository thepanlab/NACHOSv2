from typing import Dict, List, Optional, Callable
import pandas as pd

from termcolor import colored
from nachosv2.training.training_processing.training_fold import TrainingFold
from nachosv2.checkpoint_processing.read_log import read_item_list_in_log
from nachosv2.checkpoint_processing.load_save_metadata_checkpoint import write_log
from nachosv2.training.training_processing.partitions import generate_list_folds_for_partitions


def training_loop(
    execution_device: str,
    configuration: Dict,
    test_fold_name: str,
    df_metadata: pd.DataFrame,
    number_of_epochs: int,
    do_normalize_2d: bool,
    is_outer_loop: bool,
    use_mixed_precision: bool=False,
    is_3d: bool = False,
    rank: Optional[int] = None,
    is_verbose_on: bool = False
):
    """
    Creates a model, trains it, and writes its outputs.
        
    Args:
        execution_device (str): The device to be used for training.
        configuration (dict): The input configuration.
        test_fold_name (str): The name of the test fold.
        df_metadata (pandas datafram): data frame containing the metadata (filepath, label, fold, etc)
        number_of_epochs (int): The number of epochs.
        normalize_transform (bool): Whether to normalize the data.
        is_outer_loop (bool): Indicates if this is the outer loop.
        is_3d (bool): Whether the data is 3D. Default is False.
        rank (Optional[int]): The process rank. Default is None.
        is_verbose_on (bool): Indicates if verbose mode is activated. Default is False.
    """
    
    print(colored(f'Beginning the training loop for {test_fold_name}.', 'green'))
    
    
    # Gets the current rotation progress
    current_rotation = 0
    # log_rotations = read_item_list_in_log(
    #     configuration['output_path'],
    #     configuration['job_name'], 
    #     ['current_rotation'],
    #     rank
    # )
    
    
    # # If a current_rotation exists in the log, sets the next current_rotation to that value
    # if log_rotations and 'current_rotation' in log_rotations and testing_subject in log_rotations['current_rotation']:
    #     current_rotation = log_rotations['current_rotation'][testing_subject]
    #     print(colored(f'Starting off from rotation {current_rotation + 1} for testing subject {testing_subject}.', 'cyan'))
    
    
    # Trains for every current_rotation specified
    
    folds_for_partitions_list = generate_list_folds_for_partitions(
        validation_fold_list=configuration.get('validation_fold_list', None),
        fold_list=configuration['fold_list'],
        test_fold_name=test_fold_name,
        do_shuffle=configuration.get('shuffle_the_folds', False)
        ) 
    
    number_of_rotations = len(folds_for_partitions_list)
    
    for rotation in range(current_rotation, number_of_rotations):
        
        # Outer loop
        if is_outer_loop:
            validation_fold_name = None
            print(colored(f'--- Rotation {rotation + 1}/{number_of_rotations} for test subject {test_fold_name} ---', 'magenta'))
        
        # Inner loop will use the validation subjects
        else:
            validation_fold_name = folds_for_partitions_list[rotation]['validation'][0]
            print(colored(f'--- Rotation {rotation + 1}/{number_of_rotations} for test subject {test_fold_name} and val subject {validation_fold_name} ---', 'magenta'))
        
        training_folds_list = folds_for_partitions_list[rotation]['training']
        
        # Creates and runs the training fold for this subject pair
        training_fold = TrainingFold(
            execution_device,        # The name of the device that will be use
            rotation,                # The fold index within the loop
            configuration,           # The training configuration
            test_fold_name,            # The test subject name
            validation_fold_name,      # The validation_subject name
            training_folds_list,  # A list of fold partitions
            df_metadata,             # The data dictionary
            number_of_epochs,        # The number of epochs
            do_normalize_2d,         # 
            use_mixed_precision,
            rank,                    # An optional value of some MPI rank. Default is none. (Optional)
            is_outer_loop,           # If this is of the outer loop. Default is false. (Optional)
            is_3d,                   # 
            is_verbose_on            # If the verbose mode is activated. Default is false. (Optional)
        )
        
        training_fold.run_all_steps()
        
        # Writes the index to log
        write_log(
            config = configuration,
            test_fold_name = test_fold_name,
            rotation = rotation,
            log_directory = configuration['output_path'],
            log_rotations = None,
            validation_subject = None,
            rank = None,
            is_outer_loop = False
        )
        