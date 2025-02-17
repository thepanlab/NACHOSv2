from typing import Dict, List, Optional, Callable
import pandas as pd

from termcolor import colored
from nachosv2.training.training_processing.training_fold import TrainingFold
from nachosv2.checkpoint_processing.read_log import read_item_list_in_log
from nachosv2.checkpoint_processing.load_save_metadata_checkpoint import read_log
from nachosv2.checkpoint_processing.load_save_metadata_checkpoint import write_log
from nachosv2.training.training_processing.partitions import generate_list_folds_for_partitions


def training_loop(
    execution_device: str,
    configuration: Dict,
    test_fold_name: str,
    df_metadata: pd.DataFrame,
    number_of_epochs: int,
    do_normalize_2d: bool,
    is_cv_loop: bool,
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
       
    # Read checkpoint to determine rotation and rank?
    
    log_dict = read_log(configuration,
                        test_fold_name,
                        configuration['output_path'],
                        rank,
                        is_cv_loop)
    
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
    
    # if log_rotations:
    #     print(colored(f"Test fold: {log_rotations['test_fold_name']} "
    #                   f"Current rotation: {log_rotations['current_rotation']}", 'cyan'))
    
    if log_dict and test_fold_name in log_dict.keys():
        current_rotation = log_dict[test_fold_name]["rotation"]
        if log_dict[test_fold_name]["is_rotation_finished"]:
            current_rotation += 1
    else:
        current_rotation = 0
    
    # Trains for every current_rotation specified
    
    folds_for_partitions_list = generate_list_folds_for_partitions(
        validation_fold_list=configuration.get('validation_fold_list', None),
        is_cv_loop=is_cv_loop,
        fold_list=configuration['fold_list'],
        test_fold_name=test_fold_name,
        do_shuffle=configuration.get('shuffle_the_folds', False)
        ) 
    
    number_of_rotations = len(folds_for_partitions_list)
    
    for rotation in range(current_rotation, number_of_rotations):

        # Outer loop
        if is_cv_loop:
            validation_fold_name = folds_for_partitions_list[rotation]['validation'][0]
            print(colored(f'--- Rotation {rotation + 1}/{number_of_rotations} ' +
                          f'for test subject {test_fold_name} and '+
                          f'val subject {validation_fold_name} ---', 'magenta'))
        
        # Inner loop will use the validation subjects
        else:
            validation_fold_name = None
            print(colored(f'--- Rotation {rotation + 1}/{number_of_rotations} '+
                          f'for test subject {test_fold_name} ---',
                          'magenta'))

        write_log(
            config=configuration,
            test_fold=test_fold_name,
            rotation_index=rotation,
            validation_fold=validation_fold_name,
            is_rotation_finished=False,
            output_directory=configuration['output_path'],
            rank=None,
            is_cv_loop=True
        )
        
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
            is_cv_loop,           # If this is of the outer loop. Default is false. (Optional)
            is_3d,                   # 
            is_verbose_on            # If the verbose mode is activated. Default is false. (Optional)
        )
        
        training_fold.run_all_steps()
        
        write_log(
            config=configuration,
            test_fold=test_fold_name,
            rotation_index=rotation,
            validation_fold=validation_fold_name,
            is_rotation_finished=True,
            output_directory=configuration['output_path'],
            rank=None,
            is_cv_loop=False
        )
        
        # Writes the index to log
        # TODO: fix write_Log-
        # Determine if it is necessary
        # since the model it is already and other variables are already 
        # stored in the checkpoint

