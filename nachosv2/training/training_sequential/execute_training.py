from typing import List
from termcolor import colored

from nachosv2.data_processing.check_unique_subjects import check_unique_subjects
from nachosv2.data_processing.read_metadata_csv import read_metadata_csv
# from nachosv2.log_processing.read_log import read_item_list_in_log
# from nachosv2.log_processing.write_log import write_log_to_file

from nachosv2.training.training_processing.training_fold import TrainingFold
from nachosv2.checkpoint_processing.read_log import read_item_list_in_log

from nachosv2.setup.verify_configuration_types import verify_configuration_types
from nachosv2.setup.define_dimensions import define_dimensions
from nachosv2.data_processing.get_list_of_epochs import get_list_of_epochs
from nachosv2.training.training_processing.training_loop import training_loop
from nachosv2.checkpoint_processing.log_utils import write_log_to_file
from nachosv2.checkpoint_processing.load_save_metadata_checkpoint import read_log
from nachosv2.checkpoint_processing.load_save_metadata_checkpoint import write_log
from nachosv2.training.training_processing.partitions import generate_dict_folds_for_partitions

from nachosv2.training.hpo.hpo import get_hpo_configuration


def create_test_validation_pairs(test_fold_list: list,
                                 validation_fold_list: list) -> List[dict]:
    result_list = []
    for test_fold in test_fold_list:
        validation_folds = [fold for fold in validation_fold_list if fold != test_fold]
        for validation_fold in validation_folds:
            result_list.append({"test": test_fold, "validation": validation_fold})
    return result_list


def add_epochs_to_test_val_pairs(test_val_dict_list: List[dict],
                                 config_dict: dict,
                                 is_cv_loop: bool,
                                 is_verbose_on: bool) -> List[dict]:
    # verify that the number of epochs is correct
    if is_cv_loop:
        if isinstance(config_dict["hyperparameters"]["epochs"], list):
            if len(config_dict["hyperparameters"]["epochs"]) != 1:
                raise ValueError(colored("For cross-validation loop, you should have only one value for epochs.", 'red'))
            else:
                value_epoch = config_dict["hyperparameters"]["epochs"][0]
        elif isinstance(config_dict["hyperparameters"]["epochs"], int):
            value_epoch = config_dict["hyperparameters"]["epochs"]
    else:
        if len(config_dict["hyperparameters"]["epochs"]) != len(test_val_dict_list):
            raise ValueError(colored(
                f"Length of list of epochs ({len(config_dict['hyperparameters']['epochs'])}) does not match the length of test_subjects ({len(test_val_dict_list)}).",
                'red'
            ))

    test_val_epoch_list = []
            
    if is_cv_loop:
        for fold_pair in test_val_dict_list:
            fold_pair_plus = fold_pair.copy()
            fold_pair_plus["epoch"] = value_epoch
            test_val_epoch_list.append(fold_pair_plus)
    else:
        for fold_pair, epoch in zip(test_val_dict_list, config_dict["hyperparameters"]["epochs"]):
            fold_pair_plus = fold_pair.copy()
            fold_pair_plus["epoch"] = epoch
            test_val_epoch_list.append(fold_pair_plus)

    return  test_val_epoch_list
    
    
def execute_training(execution_device: str,
                     config_dict: dict,
                     is_cv_loop: bool = True,
                     is_verbose_on: bool = False):
    """
    Runs the sequential training process for each configuration and test subject.

    Args:
        execution_device (str): The name of the device that will be use.
        list_of_configs (list of JSON): The list of configuration file.
        list_of_configs_paths (list of str): The list of configuration file's paths.
        
        is_outer_loop (bool): If this is of the outer loop. (Optional)
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
    
    Raises:
        Exception: When there are repeated test subjects or validation subjects
    """

              
    # Defines if the images are 2D or 3D
    is_3d = define_dimensions(config_dict)

    # Creates the path to the data CSV files
    path_csv_metadata = config_dict["path_metadata_csv"]
    do_normalize_2d = config_dict["do_normalize_2d"]
    # Creates the normalization
    # normalizer = None
    # if not is_3d:
    #     normalizer = normalize(path_csv_metadata, dict_config)
    
    # Gets the data from the CSV files
    df_metadata = read_metadata_csv(path_csv_metadata)
    
    use_mixed_precision = config_dict["use_mixed_precision"]

    # # Reads in the log's subject list, if it exists
    # log_list = read_item_list_in_log(
    #     dict_config['output_path'],    # The directory containing the log files
    #     dict_config['job_name'],       # The prefix of the log
    #     dict_config['test_subjects'],  # The list of dictionary keys to read
    #     is_verbose_on                  # If the verbose mode is activated
    # )
    
    # # Creates the list of test subjects
    # if log_list and 'subject_list' in log_list: # From the log file if it exists
    #     test_subjects_list = log_list['test_subjects']
        
    # else: # From scratch if the log file doesn't exists
    #     test_subjects_list = dict_config['test_subjects']
        
    
    # Double-checks that the test subjects are unique
    # check_unique_subjects(test_subjects_list, "test")

    # Double-checks that the validation subjects are unique
    if is_cv_loop:  # Only if we are in the inner loop
        check_unique_subjects(config_dict["validation_fold_list"],
                                "validation")
    
    validation_fold_list = config_dict.get('validation_fold_list', None)
    
    if is_verbose_on:  # If the verbose mode is activated
        print(colored("Double-checks of test and validation uniqueness successfully done.", 'cyan'))
    
    
    # Creates test_subjects_list as a list, regardless of its initial form
    if not config_dict['test_fold_list']: # If the test_subjects list is empty, uses all subjects
        test_fold_list = list(config_dict['fold_list'])
    else:
        test_fold_list = list(config_dict['test_fold_list'])
    
    test_val_dict_list = create_test_validation_pairs(test_fold_list,
                                                      validation_fold_list)

    # test_val_epoch_dict_list = add_epochs_to_test_val_pairs(test_val_dict_list,
    #                                                         config_dict,
    #                                                         is_cv_loop,
    #                                                         is_verbose_on)

    number_training = len(test_val_dict_list)
    
    
    # Read HPO values
    hpo_configurations = get_hpo_configuration(config_dict)    
    
    for index, fold_values in enumerate(test_val_epoch_dict_list):

        test_fold = fold_values["test"]
        validation_fold = fold_values["validation"]
        n_epochs = fold_values["epoch"]

        # cross-validation loop
        if is_cv_loop:
            print(colored(f'--- Training: {index + 1}/{number_training} ---',
                          'magenta'))
        # cross-testing loop
        else:
            validation_fold_name = None
            print(colored(f'--- Training: {index + 1}/{number_training} ---',
                         'magenta'))

        print("Test fold:", test_fold)
        print("Validation folds:", validation_fold)
        print("Number of epochs:", n_epochs)
                
        partitions_dict = generate_dict_folds_for_partitions(
            validation_fold_name=validation_fold,
            is_cv_loop=is_cv_loop,
            fold_list=config_dict['fold_list'],
            test_fold_name=test_fold
            ) 
        
        write_log(
            config=config_dict,
            test_fold=test_fold,
            training_index=index,
            validation_fold=validation_fold,
            is_training_finished=False,
            output_directory=config_dict['output_path'],
            rank=None,
            is_cv_loop=True
        )
        
        training_folds_list = partitions_dict['training']
        
        # Creates and runs the training fold for this subject pair
        training_fold = TrainingFold(
            execution_device,        # The name of the device that will be use
            index,                # The fold index within the loop
            config_dict,           # The training configuration
            test_fold,            # The test subject name
            validation_fold,      # The validation_subject name
            training_folds_list,  # A list of fold partitions
            df_metadata,             # The data dictionary
            n_epochs,        # The number of epochs
            do_normalize_2d,         # 
            use_mixed_precision,
            is_cv_loop,           # If this is of the outer loop. Default is false. (Optional)
            is_3d,                   # 
            is_verbose_on            # If the verbose mode is activated. Default is false. (Optional)
        )
        
        training_fold.run_all_steps()
        
        write_log(
            config=config_dict,
            test_fold=test_fold,
            training_index=index,
            validation_fold=validation_fold,
            is_training_finished=True,
            output_directory=config_dict['output_path'],
            rank=None,
            is_cv_loop=False
        )