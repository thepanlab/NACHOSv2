from termcolor import colored

from src.data_processing.fold_generator import generate_folds


class TrainingVariables:
    def __init__(self, current_configuration, test_subject_name, is_outer_loop, is_verbose_on = False, validation_subject = None,):
        """
        Gets some parameters needed for training.
            
        Args:
            current_configuration (dict): List of input image paths.
            
            test_subject_name (str): The test subject to train.
            
            is_outer_loop (bool): A flag telling if running the outer loop.
            is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
            
            validation_subject (str): The training/validation subject. (Optional)
        """
        

        # Creates test_subjects_list as a list, regardless of its initial form
        if not current_configuration['test_subjects']: # If the test_subjects list is empty, uses all subjects
            test_subjects_list = list(current_configuration['subject_list'])
            
        else:
            test_subjects_list = list(current_configuration['test_subjects'])
        
        
        if is_outer_loop: # If outer loop, no validation
            validation_subjects_list = None
            
        else: # If inner loop
            # Creates validation_subjects_list as a list, regardless of its initial form
            validation_subjects_list = list(current_configuration['validation_subjects'])
    
    
        # Generates training folds
        self.fold_list, self.number_of_folds = generate_folds(
            test_subjects_list,
            validation_subjects_list,
            current_configuration['subject_list'],
            test_subject_name,
            current_configuration['shuffle_the_folds'],
            validation_subject = validation_subject
        )
        
        
        if is_verbose_on: # If the verbose mode is activated
            print(colored("TrainingVariables created.", 'cyan'))