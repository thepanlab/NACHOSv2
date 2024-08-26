from termcolor import colored

from scripts.training.training_sequential.sequential_subject_loop import sequential_subject_loop
from src.data_processing.check_unique_subjects import check_unique_subjects
from src.data_processing.normalize import normalize
from src.data_processing.read_data_csv import read_data_csv
from src.log_processing.read_log import read_item_list_in_log
from src.log_processing.write_log import write_log_to_file
from src.setup.check_configuration_types import check_configuration_types
from src.setup.define_dimensions import define_dimensions
from src.data_processing.get_list_of_epochs import get_list_of_epochs


def sequential_processing(execution_device, list_of_configs, list_of_configs_paths, is_outer_loop, is_verbose_on = False):
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

    # Does the sequential training process for each configuration file
    for configuration_iterator in range(len(list_of_configs)):
        
        current_configuration = list_of_configs[configuration_iterator]
        current_configuration_path = list_of_configs_paths[configuration_iterator]
        
        # Checks the arguments' type of the current configuration file
        check_configuration_types(current_configuration, current_configuration_path)


        # Defines if the images are 2D or 3D
        is_3d = define_dimensions(current_configuration)


        # Creates the path to the data CSV files
        
        if is_3d:
            data_path = "data/3D_kidney_csv" # TODO
        
        else:
            data_path = "data/2D_kidney_csv" # TODO
        
        # Creates the normalization
        normalize_transform = None
        if not is_3d:
            normalize_transform = normalize(data_path, current_configuration)
        
        # Gets the data from the CSV files
        data_dictionary = read_data_csv(data_path, current_configuration)

        
        # Reads in the log's subject list, if it exists
        log_list = read_item_list_in_log(
            current_configuration['output_path'],   # The directory containing the log files
            current_configuration['job_name'],      # The prefix of the log
            ['test_subjects'],                      # The list of dictionary keys to read
            is_verbose_on                           # If the verbose mode is activated
        )
        
        
        # Creates the list of test subjects
        if log_list and 'subject_list' in log_list: # From the log file if it exists
            test_subjects_list = log_list['test_subjects']
            
        else: # From scratch if the log file doesn't exists
            test_subjects_list = current_configuration['test_subjects']
            
        
        # Double-checks that the test subjects are unique
        check_unique_subjects(test_subjects_list, "test")

        # Double-checks that the validation subjects are unique
        if is_outer_loop == False: # Only if we are in the inner loop
            check_unique_subjects(current_configuration["validation_subjects"], "validation")
        
        if is_verbose_on: # If the verbose mode is activated
            print(colored("Double-checks of test and validation uniqueness successfully done.", 'cyan'))
        
        
        # Creates test_subjects_list as a list, regardless of its initial form
        if not current_configuration['test_subjects']: # If the test_subjects list is empty, uses all subjects
            test_subjects_list = list(current_configuration['subject_list'])
            
        else:
            test_subjects_list = list(current_configuration['test_subjects'])
        
        
        # Gets the list of epochs
        list_of_epochs = get_list_of_epochs(current_configuration["hyperparameters"]["epochs"], test_subjects_list, is_outer_loop, is_verbose_on)
        
        
        # Trains for each test subject
        for test_subject_name, number_of_epochs in zip(test_subjects_list, list_of_epochs):

            sequential_subject_loop(
                execution_device,       # The name of the device that will be use
                current_configuration,  # The training configuration
                test_subject_name,      # The test subject name
                data_dictionary,        # The data dictionary
                number_of_epochs,       # The number of epochs
                normalize_transform,    # The normalization transformation
                is_outer_loop,          # If this is of the outer loop
                is_3d,
                is_verbose_on           # If the verbose mode is activated
            )
            
            
            test_subjects_dictionary = {'test_subjects': [t for t in test_subjects_list if t != test_subject_name]}
            
            write_log_to_file(
                current_configuration['output_path'],   # The directory containing the log files
                current_configuration['job_name'],      # The prefix of the log
                test_subjects_dictionary,               # A dictionary of the relevant status info
                False, # use_lock                       # whether to use a lock or not when writing results
                is_verbose_on                           # If the verbose mode is activated
            )
    