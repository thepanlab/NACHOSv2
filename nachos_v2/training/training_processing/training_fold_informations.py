from termcolor import colored

import torch.nn as nn
import torch.optim as optim

from src.log_processing.checkpointer import *
from src.image_processing.image_crop import create_crop_box
from src.image_processing.image_parser import *
from src.log_processing.delete_log import *
from src.model_processing.create_model import create_training_model
from src.model_processing.get_metrics_dictionary import get_metrics_dictionary
from src.model_processing.initialize_model_weights import initialize_model_weights
from src.modules.early_stopping.early_stopping import create_early_stopping
from src.modules.optimizer.optimizer_creator import create_optimizer


class _TrainingFoldInformations():
    def __init__(self, fold_index, current_configuration, testing_subject, validation_subject, fold_list, data_dictionary, number_of_epochs, normalize_transform, mpi_rank = None, is_outer_loop = False, is_3d = False, is_verbose_on = False):
        """
        Initializes a training fold informations object.

        Args:
            fold_index (int): The fold index within the loop.
            current_configuration (dict): The training configuration.
            
            testing_subject (str): The test subject name.
            validation_subject (str): The validation subject name.
            
            fold_list (list of dict): A list of fold partitions.
            data_dictionary (dict of list): The data dictionary.
            number_of_epochs (int): The number of epochs.
                      
            mpi_rank (int): An optional value of some MPI rank. Default is none. (Optional)
            is_outer_loop (bool): If this is of the outer loop. Default is false. (Optional)
            is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
        """ 

        # Initializations
        self.fold_index = fold_index
        self.current_configuration = current_configuration
        self.hyperparameters = current_configuration['hyperparameters']
        
        self.testing_subject = testing_subject
        self.validation_subject = validation_subject
        
        self.fold_list = fold_list
        self.data_dictionary = data_dictionary
        self.number_of_epochs = number_of_epochs
        
        self.metrics_dictionary = get_metrics_dictionary(self.current_configuration['metrics'])
        
        self.mpi_rank = mpi_rank
        self.is_outer_loop = is_outer_loop
        self.is_3d = is_3d
        
        self.datasets_dictionary = { # Dictionary of dictionaries
            'testing':    {'files': [], 'indexes': [], 'labels': [], 'ds': None},
            'training':   {'files': [], 'indexes': [], 'labels': [], 'ds': None},
            'validation': {'files': [], 'indexes': [], 'labels': [], 'ds': None}
        }
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.list_callbacks = None
        self.loss_function = nn.CrossEntropyLoss()

        self.normalization = normalize_transform
        self.crop_box = create_crop_box(
            self.hyperparameters['cropping_position'][0], # The height of the offset
            self.hyperparameters['cropping_position'][1], # The width of the offset
            self.current_configuration['target_height'],  # The height wanted
            self.current_configuration['target_width'],   # The width wanted
            is_verbose_on
        )
        
        
        # If MPI, specifies the job name by task
        if self.mpi_rank:
            
            base_job_name = current_configuration['job_name'] # Only to put in the new job name
            
            # Checks if inner or outer loop
            if is_outer_loop: # Outer loop => no validation
                new_job_name = f"{base_job_name}_test_{testing_subject}"
                
            else: # Inner loop => validation
                new_job_name = f"{base_job_name}_test_{testing_subject}_val_{validation_subject}"

            # Updates the job name
            self.current_configuration['job_name'] = new_job_name
        
        
        if is_verbose_on: # If the verbose mode is activated
            print(colored("Fold of training informations created.", 'cyan'))


        
    def _run_all_steps(self):
        """
        Runs all of the steps in order to create the basic training information.
        """
        
        self.get_dataset_info()
        self.create_model()
        self.optimizer = create_optimizer(self.model, self.current_configuration['hyperparameters'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', factor = 0.1, patience = self.current_configuration['hyperparameters']['patience'], verbose = True)
        self.create_callbacks()



    def get_dataset_info(self):
        """
        Gets the basic data to create the datasets from.
        This includes the testing and validation_subjects, file paths,
        label indexes, and labels.
        """

        # Initialization
        total_index = 0
        
        # Loops through all image files and determines which dataset they belong to
        for subject_name in self.data_dictionary:
             
            dataset_type = self.get_dataset_type(subject_name)

            # Appends dataset item to the dictionary
            for index in self.data_dictionary[subject_name]['indexes']:
            
                self.datasets_dictionary[dataset_type]['files'].append(self.data_dictionary[subject_name]['image_path'][index]) # The file name
                self.datasets_dictionary[dataset_type]['indexes'].append(total_index)                                           # The index in the dictionary's lists
                self.datasets_dictionary[dataset_type]['labels'].append(self.data_dictionary[subject_name]['labels'][index])    # The label number

                total_index += 1
    
    
    
    def get_dataset_type(self, subject_name):
        """
        Gets the right dataset type for ths given subject.
        
        Args:
            subject_name (str): The subject.
            
        Returns:
            dataset_type (str): The dataset type.
        """
        
        # Initialization
        dataset_type = ''
        
        
        # Sets the right dataset_type
        if subject_name == self.testing_subject:
            dataset_type = 'testing'
            
        elif not self.is_outer_loop and subject_name == self.validation_subject: # Only for the inner loop
            dataset_type = 'validation'
            
        elif subject_name in self.current_configuration['subject_list']:
            dataset_type = 'training'
        
        
        return dataset_type
    
    

    def create_model(self):
        """
        Creates the initial model for training and initializes its weights.
        """
        
        # Creates the model
        self.model = create_training_model(self.current_configuration)
        
        
        # Initializes the weights
        initialize_model_weights(self.model)
    
    
    
    def create_callbacks(self):
        """
        Creates the training callbacks. 
        This includes early stopping and checkpoints.
        """
        
        # Gets the job name to create the checkpoint prefix
        if self.is_outer_loop:          
            self.checkpoint_prefix = f"{self.current_configuration['job_name']}_test_{self.testing_subject}_config_{self.current_configuration['selected_model_name']}"
            
        else:
            self.checkpoint_prefix = f"{self.current_configuration['job_name']}_test_{self.testing_subject}_val_{self.validation_subject}_config_{self.current_configuration['selected_model_name']}"
        
        # Creates the path where to save the checkpoint
        save_path = os.path.join(self.current_configuration['output_path'], 'checkpoints')
        
        # Creates the checkpoint
        checkpointer = Checkpointer(
            self.number_of_epochs,
            self.current_configuration['k_epoch_checkpoint_frequency'],
            self.checkpoint_prefix,
            self.mpi_rank,
            save_path
        )
        
        
        # Early stopping
        if not self.is_outer_loop: # Checks if not outer loop for early stopping
            early_stopping = create_early_stopping(self.optimizer, self.hyperparameters['patience'])
            
            self.list_callbacks = [early_stopping, checkpointer]
            
        else:
            self.list_callbacks = [checkpointer]
