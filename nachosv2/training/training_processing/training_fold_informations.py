from typing import Dict
from collections import OrderedDict

import pandas as pd
from termcolor import colored
from typing import Optional, Callable

import torch.nn as nn
import torch.optim as optim

from nachosv2.checkpoint_processing.checkpointer import *
from nachosv2.image_processing.image_crop import create_crop_box
from nachosv2.image_processing.image_parser import *
from nachosv2.checkpoint_processing.delete_log import *
from nachosv2.model_processing.create_model import create_model
from nachosv2.model_processing.get_metrics_dictionary import get_metrics_dictionary
from nachosv2.model_processing.initialize_model_weights import initialize_model_weights
from nachosv2.modules.early_stopping.early_stopping import create_early_stopping
from nachosv2.modules.optimizer.optimizer_creator import create_optimizer
from nachosv2.data_processing.normalizer import normalizer


class TrainingFoldInformations():
    def __init__(
        self,
        rotation_index: int,
        configuration: dict,
        test_fold_name: str,
        validation_fold_name: str,
        training_folds_list: list[dict],
        df_metadata: pd.DataFrame,
        number_of_epochs: int,
        do_normalize_2d: bool=False,
        mpi_rank: int = None,
        is_outer_loop: bool = False,
        is_3d: bool = False,
        is_verbose_on: bool = False
    ):
        """
        Initializes a training fold informations object.

        Args:
            fold_index (int): The fold index within the loop.
            configuration (dict): The training configuration.
            testing_subject (str): The test subject name.
            validation_subject (str): The validation subject name.
            fold_list (list of dict): A list of fold partitions.
            data_dictionary (dict of list): The data dictionary.
            number_of_epochs (int): The number of epochs.
            normalize_transform (bool): Whether to normalize the data. Default is False. (Optional)     
            mpi_rank (int): An optional value of some MPI rank. Default is none. (Optional)
            is_outer_loop (bool): If this is of the outer loop. Default is false. (Optional)
            is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
        """ 

        # Initializations
        self.rotation_index = rotation_index
        self.configuration = configuration
        self.hyperparameters = configuration['hyperparameters']
        
        self.test_fold_name = test_fold_name
        self.validation_fold_name = validation_fold_name
        self.training_folds_list = training_folds_list
        
        self.df_metadata = df_metadata
        self.number_of_epochs = number_of_epochs
        self.metrics_dictionary = get_metrics_dictionary(self.configuration['metrics_list'])
        
        self.mpi_rank = mpi_rank
        self.is_outer_loop = is_outer_loop
        self.is_3d = is_3d
        
        self.dictionary_partitions_info = OrderedDict( # Dictionary of dictionaries
            [('training', {'files': [], 'labels': [], 'ds': None}),
             ('validation', {'files': [], 'labels': [], 'ds': None}),
             ('test', {'files': [], 'labels': [], 'ds': None}),
            ]
            )
        
        # self.list_callbacks = None
        # self.loss_function = nn.CrossEntropyLoss()

        # self.do_normalize_2d = do_normalize_2d
        # self.crop_box = create_crop_box(
        #     self.hyperparameters['cropping_position'][0], # The height of the offset
        #     self.hyperparameters['cropping_position'][1], # The width of the offset
        #     self.configuration['target_height'],  # The height wanted
        #     self.configuration['target_width'],   # The width wanted
        #     is_verbose_on
        # )
        
        # If MPI, specifies the job name by task
        # if self.mpi_rank:
        #     base_job_name = configuration['job_name'] # Only to put in the new job name
            
        #     # Checks if inner or outer loop
        #     if is_outer_loop: # Outer loop => no validation
        #         new_job_name = f"{base_job_name}_test_{test_subject}"
                
        #     else: # Inner loop => validation
        #         new_job_name = f"{base_job_name}_test_{test_subject}_val_{validation_subject}"

        #     # Updates the job name
        #     self.configuration['job_name'] = new_job_name
            
        if is_verbose_on:  # If the verbose mode is activated
            print(colored("Fold of training informations created.", 'cyan'))

     
    # def run_all_steps(self):
    #     """
    #     Runs all of the steps in order to create the basic training information.
    #     """
        
    #     self.get_dataset_info()
    #     self.create_model()
    #     self.optimizer = create_optimizer(self.model, self.configuration['hyperparameters'])
    #     self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', factor = 0.1, patience = self.configuration['hyperparameters']['patience'], verbose = True)
    #     self.create_callbacks()


    # def get_normalizer(self):
    #     normalizer = None
    #     if not is_3d:
    #         normalizer = normalize(self.dictionary_partitions_info["training"]["files"],
    #                                dict_config)


    # def get_dataset_info(self):
    #     """
    #     Gets the basic data to create the datasets from.
    #     This includes the testing and validation_subjects, file paths,
    #     label indexes, and labels.
    #     """
    #     # TODO
    #     # Add option to choose the name of the column for subject and filepath

    #     l_columns = ["absolute_filepath", "label"]
        
    #     # verify values in l_columns are in df_metadata.columns
    #     if not all(val_col in self.df_metadata.columns.to_list() for val_col in l_columns):
    #         raise ValueError(f"The columns {l_columns} must be in csv_metadata file.")
        
    #     for partition, value_dict in self.dictionary_partitions_info.items():
    #         if partition == 'testing':
    #             value_dict['files'] = \
    #                 self.df_metadata[self.df_metadata["subject"] == self.test_subject]["absolute_filepath"].tolist()
    #             value_dict['labels'] = \
    #                 self.df_metadata[self.df_metadata["subject"] == self.test_subject]["label"].tolist()
    #         elif partition == 'validation':
    #             value_dict['files'] = \
    #                 self.df_metadata[self.df_metadata["subject"] == self.validation_subject]["absolute_filepath"].tolist()
    #             value_dict['labels'] = \
    #                 self.df_metadata[self.df_metadata["subject"] == self.validation_subject]["label"].tolist()
    #         else:  # training
    #             value_dict['files'] = \
    #                 self.df_metadata[self.df_metadata["subject"].isin(self.list_training_subjects)]["absolute_filepath"].tolist()
    #             value_dict['labels'] = \
    #                 self.df_metadata[self.df_metadata["subject"].isin(self.list_training_subjects)]["label"].tolist()  
    
    
    # def get_dataset_type(self, subject_name):
    #     """
    #     Gets the right dataset type for ths given subject.
        
    #     Args:
    #         subject_name (str): The subject.
            
    #     Returns:
    #         dataset_type (str): The dataset type.
    #     """
        
    #     # Initialization
    #     dataset_type = ''
        
    #     # Sets the right dataset_type
    #     if subject_name == self.testing_subject:
    #         dataset_type = 'testing'
            
    #     elif not self.is_outer_loop and subject_name == self.validation_subject: # Only for the inner loop
    #         dataset_type = 'validation'
            
    #     elif subject_name in self.configuration['subject_list']:
    #         dataset_type = 'training'
        
    #     return dataset_type
    
    
    # def create_model(self):
    #     """
    #     Creates the initial model for training and initializes its weights.
    #     """
        
    #     # Creates the model
    #     self.model = create_training_model(self.configuration)
        
    #     # Initializes the weights
    #     initialize_model_weights(self.model)
    

    # def create_callbacks(self):
    #     """
    #     Creates the training callbacks. 
    #     This includes early stopping and checkpoints.
    #     """
        
    #     # Gets the job name to create the checkpoint prefix
    #     if self.is_outer_loop:          
    #         self.checkpoint_prefix = f"{self.configuration['job_name']}_test_{self.test_subject}_config_{self.configuration['selected_model_name']}"
            
    #     else:
    #         self.checkpoint_prefix = f"{self.configuration['job_name']}_test_{self.test_subject}_val_{self.validation_subject}_config_{self.configuration['selected_model_name']}"
        
    #     # Creates the path where to save the checkpoint
    #     save_path = os.path.join(self.configuration['output_path'], 'checkpoints')
        
    #     # Creates the checkpoint
    #     checkpointer = Checkpointer(
    #         self.number_of_epochs,
    #         self.configuration['k_epoch_checkpoint_frequency'],
    #         self.checkpoint_prefix,
    #         self.mpi_rank,
    #         save_path
    #     )
        
        
    #     # Early stopping
    #     if not self.is_outer_loop: # Checks if not outer loop for early stopping
    #         early_stopping = create_early_stopping(self.optimizer, self.hyperparameters['patience'])
            
    #         self.list_callbacks = [early_stopping, checkpointer]
            
    #     else:
    #         self.list_callbacks = [checkpointer]
