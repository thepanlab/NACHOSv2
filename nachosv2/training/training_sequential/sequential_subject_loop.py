from typing import Dict, Optional, Callable

from termcolor import colored
import pandas as pd

from nachosv2.training.training_processing.partitions import generate_list_subjects_for_partitions
from nachosv2.training.training_processing.training_loop import training_loop


def sequential_subject_loop(
    execution_device: str,
    current_configuration: Dict,
    test_subject: str,
    df_metadata: pd.DataFrame,
    number_of_epochs: int,
    do_normalize_2d: bool=False,
    is_outer_loop: bool=False,
    is_3d: bool = False,
    is_verbose_on: bool = False
):
    """
    Executes the training loop for the given test subject.

    Args:
        execution_device (str): The name of the device to be used for training.
        current_configuration (dict): The training configuration.
        test_subject_name (str): The name of the test subject.
        df_metadata (pd.DataFrame): Dataframe containing filename, subject, and other metadata.
        number_of_epochs (int): The number of epochs for training.
        normalize_transform (bool): Whether to normalize the data.
        is_outer_loop (bool): Whether this is an outer loop run (no validation subjects).
        is_3d (bool): Whether the data is 3D. Default is False.
        is_verbose_on (bool): Whether verbose mode is activated. Default is False.
    """
    
    print(colored(
        "\n========== " + 
        f"Starting training for {test_subject} in {current_configuration['selected_model_name']}" +
        " ==========\n",
        'green'))

    # Gets the needed data for training
    
    list_subjects_for_partitions = generate_list_subjects_for_partitions(
        list_validation_subjects=current_configuration.get('validation_subjects', None),
        list_subjects=current_configuration['subject_list'],
        test_subject=test_subject,
        do_shuffle=current_configuration.get('shuffle_the_folds', False)
        )    

    # Starts the training loop
    
    training_loop(
        execution_device=execution_device,
        configuration=current_configuration,
        test_subject=test_subject,
        list_subjects_for_partitions=list_subjects_for_partitions,
        df_metadata=df_metadata,
        number_of_epochs=number_of_epochs,
        do_normalize_2d=do_normalize_2d,
        is_outer_loop=is_outer_loop,
        is_3d=is_3d,
        is_verbose_on=is_verbose_on
    )