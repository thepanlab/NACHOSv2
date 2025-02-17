import random
import os
from collections import OrderedDict
from typing import Optional, Callable, Union, Tuple, List
from pathlib import Path
import math

from termcolor import colored
import pandas as pd
from datetime import datetime
# import torch
import torch
# from torch.cuda.amp import autocast, GradScaler
from torch import autocast, GradScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

from nachosv2.training.training_processing.custom_2D_dataset import Dataset2D
# from nachosv2.training.training_processing.custom_3D_dataset import Custom3DDataset
from nachosv2.training.training_processing.training_fold_informations import TrainingFoldInformations
from nachosv2.data_processing.create_empty_history import create_empty_history
from nachosv2.data_processing.normalizer import normalizer
from nachosv2.image_processing.image_crop import create_crop_box
from nachosv2.image_processing.image_parser import *
from nachosv2.checkpoint_processing.checkpointer import Checkpointer
from nachosv2.checkpoint_processing.delete_log import delete_log_file
from nachosv2.checkpoint_processing.read_log import read_item_list_in_log
from nachosv2.checkpoint_processing.load_save_metadata_checkpoint import save_metadata_checkpoint
# from nachosv2.model_processing.save_model import save_model
from nachosv2.modules.timer.precision_timer import PrecisionTimer
from nachosv2.output_processing.result_outputter import predict_and_save_results
from nachosv2.results_processing.class_recall.epoch_recall import epoch_class_recall
from nachosv2.model_processing.initialize_model_weights import initialize_model_weights        
from nachosv2.model_processing.create_model import create_model
from nachosv2.model_processing.get_metrics_dictionary import get_metrics_dictionary
from nachosv2.modules.optimizer.optimizer_creator import create_optimizer
from typing import Union, List
from nachosv2.modules.early_stopping.earlystopping import EarlyStopping
from nachosv2.output_processing.result_outputter import save_history_to_csv

class TrainingFold():
    def __init__(
        self,
        execution_device: str,
        rotation_index: int,
        configuration: dict,
        test_fold_name: str,
        validation_fold_name: str,
        training_folds_list: List[str],
        df_metadata: pd.DataFrame,
        number_of_epochs: int,
        do_normalize_2d: bool = False,
        use_mixed_precision: bool = False,
        mpi_rank: int = None,
        is_cv_loop: bool = False,
        is_3d: bool = False,
        is_verbose_on: bool = False
    ):
        """ Initializes a training fold object.

        Args:
            execution_device (str): The name of the device that will be use.
            rotation_index (int): The fold index within the loop.
            configuration (dict): The training configuration.
            
            testing_subject (str): The test subject name.
            validation_subject (str): The validation_subject name.
            
            fold_list (list of dict): A list of fold partitions.
            
            data_dictionary (dict of list): The data dictionary.
            number_of_epochs (int): The number of epochs.
                    
            mpi_rank (int): An optional value of some MPI rank. Default is none. (Optional)
            is_outer_loop (bool): If this is of the outer loop. Default is false. (Optional)
            is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
        """
            
        self.fold_info = TrainingFoldInformations(
            rotation_index,         # The fold index within the loop
            configuration,          # The training configuration
            test_fold_name,         # The test subject name
            validation_fold_name,   # The validation subject name
            training_folds_list,    # A list of fold partitions
            df_metadata,            # The data dictionary
            number_of_epochs,       # The number of epochs
            do_normalize_2d,        #
            mpi_rank,               # An optional value of some MPI rank
            is_cv_loop,          # If this is of the outer loop
            is_3d,                  #
            is_verbose_on           # If the verbose mode is activated
        )
        
        self.rotation_index = rotation_index
        self.configuration = configuration
        self.hyperparameters = configuration['hyperparameters']
        
        self.test_fold_name = test_fold_name
        self.validation_fold_name = validation_fold_name
        
        if is_cv_loop:
            self.prefix_name = f"{configuration['job_name']}" + \
                               f"_{test_fold_name}"
        else:
            self.prefix_name = f"{configuration['job_name']}" + \
                               f"_{test_fold_name}_{validation_fold_name}"

        self.training_folds_list = training_folds_list
        
        self.df_metadata = df_metadata
        self.number_of_epochs = number_of_epochs
        self.metrics_dictionary = get_metrics_dictionary(self.configuration['metrics_list'])
        
        self.partitions_info_dict = OrderedDict( # Dictionary of dictionaries
            [('training', {'files': [], 'labels': [], 'dataloader': None}),
             ('validation', {'files': [], 'labels': [], 'dataloader': None}),
             ('test', {'files': [], 'labels': [], 'dataloader': None}),
            ]
            )
        
        self.list_callbacks = None
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # The input is expected to contain the unnormalized logits for each class
        self.loss_function = nn.CrossEntropyLoss()
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        self.execution_device = execution_device
        
        self.is_cv_loop = is_cv_loop
        
        self.history = create_empty_history(self.is_cv_loop, 
                                            self.metrics_dictionary)
        self.time_elapsed = None
        
        self.counter_early_stopping = 0
        
        self.partitions_info_dict = OrderedDict( # Dictionary of dictionaries
            [('training', {'files': [], 'labels': [], 'dataloader': None}),
             ('validation', {'files': [], 'labels': [], 'dataloader': None}),
             ('test', {'files': [], 'labels': [], 'dataloader': None}),
            ]
            )
        
        self.use_mixed_precision = use_mixed_precision
        self.mpi_rank = mpi_rank

        self.is_3d = is_3d
        self.do_normalize_2d = do_normalize_2d
        
        self.crop_box = create_crop_box(
            self.hyperparameters['cropping_position'][0], # The height of the offset
            self.hyperparameters['cropping_position'][1], # The width of the offset
            self.configuration['target_height'],  # The height wanted
            self.configuration['target_width'],   # The width wanted
            is_verbose_on
        )
        
        self.checkpoint_folder_path = Path(self.configuration['output_path']) / 'checkpoints'
        self.start_epoch = 0
        
        self.prev_checkpoint_file_path = None
        self.prev_best_checkpoint_file_path = None
        
        # If MPI, specifies the job name by task
        if self.mpi_rank:
            job_name = configuration['job_name'] # Only to put in the new job name
            
            # Checks if inner or outer loop
            if is_cv_loop: # Outer loop => no validation
                new_job_name = f"{job_name}_test_{test_fold_name}" 
            else: # Inner loop => validation
                new_job_name = f"{job_name}_test_{test_fold_name}"+ \
                               f"_val_{validation_fold_name}"

            # Updates the job name
            self.configuration['job_name'] = new_job_name


    # def create_callbacks(self):
    #     """
    #     Creates the training callbacks. 
    #     This includes early stopping and checkpoints.
    #     """
        
    #     # Gets the job name to create the checkpoint prefix
    #     if self.is_outer_loop:          
    #         self.checkpoint_prefix = f"{self.configuration['job_name']}_test_{self.test_fold_name}" + \
    #                                  f"_config_{self.configuration['architecture_name']}"
            
    #     else:
    #         self.checkpoint_prefix = f"{self.configuration['job_name']}_test_{self.test_fold_name}" + \
    #                                  f"_val_{self.validation_subject}_config_{self.configuration['architecture_name']}"
        
    #     # Creates the path where to save the checkpoint
    #     self.save_path = Path(self.configuration['output_path']) / 'checkpoints'
        
    #     # Creates the checkpoint
    #     checkpointer = Checkpointer(
    #         self.number_of_epochs,
    #         self.configuration['k_epoch_checkpoint_frequency'],
    #         self.checkpoint_prefix,
    #         self.mpi_rank,
    #         save_path
    #     )


    def get_normalizer(self):
        normalizer = None
        if not self.is_3d:
            normalizer = normalizer(self.partitions_info_dict["training"]["files"],
                                    dict_config)


    def get_dataset_info(self):
        """
        Gets the basic data to create the datasets from.
        This includes the testing and validation_subjects, file paths,
        label indexes, and labels.
        """
        # TODO
        # Add option to choose the name of the column for subject and filepath

        l_columns = ["absolute_filepath", "label"]
        
        # verify values in l_columns are in df_metadata.columns
        if not all(val_col in self.df_metadata.columns.to_list() for val_col in l_columns):
            raise ValueError(f"The columns {l_columns} must be in csv_metadata file.")
        
        for partition, value_dict in self.partitions_info_dict.items():
            if partition == 'test':
                are_test_rows_series_bool = self.df_metadata["fold_name"] == self.test_fold_name
                value_dict['files'] = \
                    self.df_metadata[are_test_rows_series_bool]["absolute_filepath"].tolist()
                value_dict['labels'] = \
                    self.df_metadata[are_test_rows_series_bool]["label"].tolist()
            elif partition == 'validation':
                are_validation_rows_series_bool = self.df_metadata["fold_name"] == self.validation_fold_name
                value_dict['files'] = \
                    self.df_metadata[are_validation_rows_series_bool]["absolute_filepath"].tolist()
                value_dict['labels'] = \
                    self.df_metadata[are_validation_rows_series_bool]["label"].tolist()
            else:  # training
                are_training_rows_series_bool = self.df_metadata["fold_name"].isin(self.training_folds_list)
                value_dict['files'] = \
                    self.df_metadata[are_training_rows_series_bool]["absolute_filepath"].tolist()
                value_dict['labels'] = \
                    self.df_metadata[are_training_rows_series_bool]["label"].tolist()  


    def create_model(self):
        """
        Creates the initial model for training and initializes its weights.
        """
        
        # Creates the model
        self.model = create_model(self.configuration)
        self.model.to(self.execution_device)
        # Initializes the weights
        # initialize_model_weights(self.model)
        self.model.apply(initialize_model_weights)


    def run_all_steps(self):
        """
        Runs all of the steps for the training process.
        Training itself depends on the state of the training fold.
        Checks for insufficient dataset.
        """
        
        # Loads in the previously saved fold info. Check if valid. If so, use it.
        # prev_info = self.load_state()

        # if prev_info is not None and \
        # self.fold_info.testing_subject == prev_info.testing_subject and \
        # self.fold_info.validation_subject == prev_info.validation_subject:
        #     self.fold_info = prev_info
        #     self.load_checkpoint()
        #     # Conditional if there is no checkpoint
            
        #     print(colored("Loaded previous existing state for testing subject " + 
        #                   f"{prev_info.testing_subject} and subject {prev_info.validation_subject}.", 'cyan'))

        # # Computes every thing if no checkpoint
        # else:
        #     self.fold_info.state()
                
       # Replace for all substeps
                
        self.get_dataset_info()
        self.create_model()
        self.optimizer = create_optimizer(self.model,
                                          self.configuration['hyperparameters'])
        
        # scheduler https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

        # eta_min=self.configuration['hyperparameters']['learning_rate']/10,        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                              T_max=10,
                                                              eta_min=self.configuration['hyperparameters']['learning_rate']/100,
                                                              last_epoch=-1,
                                                              verbose='deprecated')
        
        # self.save_state()
            
        # Creates the datasets and trains them (Datasets cannot be logged.)
        if self.create_dataset():
            self.train_model()
            # TODO reorganize and rename functions to make it more understandable
            self.process_results()


    def load_state(self):
        """
        Loads the latest training state.
        """
        
        # Loads the log file
        log = read_item_list_in_log(
            self.configuration['output_path'],
            self.configuration['job_name'], 
            ['fold_info']
        )
        
        # If there is no log or on fold_info in log, returns None
        if log is None or 'fold_info' not in log:
            logged_fold_info = None
        
        # If there is log, returns it
        else:
            logged_fold_info = log['fold_info']
        
        return logged_fold_info


    def save_state(self):
        """
        Saves the state of the fold to a log.
        """
       
        save_metadata_checkpoint(
            output_directory=self.configuration['output_path'],
            prefix_filename=self.configuration['job_name'],
            dict_to_save={'fold_info': self.fold_info},
            use_lock=self.mpi_rank != None
        )


    # def load_checkpoint(self):
    #     """
    #     Loads the latest checkpoint to start from.
    #     """
        
    #     checkpoint = get_most_recent_checkpoint(
    #         os.path.join(self.fold_info.configuration['output_path'], 'checkpoints'),
    #         self.fold_info.checkpoint_prefix,
    #         self.fold_info.model
    #     )
        
    #     if checkpoint is not None:
    #         print(colored(f"Loaded most recent checkpoint of epoch: {checkpoint[1]}.", 'cyan'))
    #         self.fold_info.model.model = checkpoint[0]
    #         self.checkpoint_epoch = checkpoint[1]
    #     # If no checkpoint
    #     else:
    #         # Creates the model
    #         self.fold_info.create_model()
    
    
    float_or_list_floats = Union[float, List[float]]
    def get_mean_stddev(self, dataloader: DataLoader)->Tuple[float_or_list_floats,
                                                             float_or_list_floats]:
        """
        Calculate the mean and standard deviation for grayscale (1 channel) or RGB (3 channels) datasets.

        Args:
            dataloader (DataLoader): DataLoader providing batches of images.
            
        Returns:
            Tuple[Union[float, List[float]], Union[float, List[float]]]: 
            (mean, std), where mean and std are scalars (float) for grayscale 
            or lists of floats for RGB.
        """

        total_sum = torch.zeros(self.hyperparameters["number_channels"])
        total_sum_squared = torch.zeros(self.hyperparameters["number_channels"])
        num_pixels = 0
        
        # Iterate through the dataset
        for images, _, _ in dataloader:
            batch_size, channels, height, width = images.shape

            images = images.view(batch_size, channels, -1)
            
            # Accumulate sum and squared sum
            total_sum += images.sum(dim=(0, 2))  # Sum over batch and spatial dimensions
            total_sum_squared += (images ** 2).sum(dim=(0, 2))  # Sum of squares
            num_pixels += batch_size * height * width  # Total number of pixels per channel

        # Calculate mean and standard deviation
        mean = total_sum / num_pixels
        stddev = (total_sum_squared / num_pixels - mean ** 2).sqrt()
            
        # Convert to scalars for grayscale or lists for RGB
        if self.hyperparameters["number_channels"] == 1:
            return mean.item(), stddev.item()
        elif self.hyperparameters["number_channels"] == 3:
            return mean.tolist(), stddev.tolist()
        else:
            raise ValueError("Number of channels must be 1 or 3.")
        
        
    def create_dataset(self) -> bool:
        """
        Creates the dataset needed to trains the model.
        It will map the image paths into their respective image and label pairs.
        TODO Complete any image-reading changes here for different file types.
        
        Returns:
            _is_dataset_created (bool): If the datasets are created or not.
        """
        
        # Gets the datasets for each phase
        for dataset_type in self.partitions_info_dict:
            
            # If there is no files for the current dataset (= no validation = outer loop)
            if not self.partitions_info_dict[dataset_type]['files']:
                continue  # Skips the rest of the loop
            
            drop_residual = False
            # If the current dataset is the training one
            if dataset_type == "training":
                
                # Calculates the residual
                drop_residual = self._residual_compute_and_decide()
                
                transform = None
                
                if self.do_normalize_2d:
                    dataset_before_normalization = Dataset2D(
                        dictionary_partition = self.partitions_info_dict[dataset_type],
                        number_channels = self.hyperparameters['number_channels'],
                        do_cropping = self.hyperparameters['do_cropping'],
                        crop_box = self.crop_box,
                        transform=None
                    )

                    dataloader = DataLoader(
                        dataset = dataset_before_normalization,
                        batch_size = self.configuration['hyperparameters']['batch_size'],
                        shuffle = False,
                        drop_last = drop_residual,
                        num_workers = 0 # Default
                    )
                    
                    mean, stddev = self.get_mean_stddev(dataloader=dataloader)
                    transform = transforms.Normalize(mean=mean, std=stddev)

            # Creates the dataset
            if self.is_3d:
                dataset = Custom3DDataset(
                    self.configuration['data_input_directory'],   # The file path prefix = the path to the directory where the images are
                    self.partitions_info_dict[dataset_type],               # The data dictionary
                )
                
            else:
                dataset = Dataset2D(
                    dictionary_partition = self.partitions_info_dict[dataset_type], # The data dictionary
                    number_channels = self.hyperparameters['number_channels'],      # The number of channels in an image
                    do_cropping = self.hyperparameters['do_cropping'],              # Whether to crop the image
                    crop_box = self.crop_box,                                       # The dimensions after cropping
                    transform=transform
                )

            # TODO verify the shuffle           
            # Shuffles the images
            dataset, do_shuffle = self._shuffle_dataset(dataset, dataset_type)

            # Apply normalization to the dataset

            # Creates dataloader
            dataloader = DataLoader(
                dataset = dataset,
                batch_size = self.configuration['hyperparameters']['batch_size'],
                shuffle = do_shuffle,
                drop_last = drop_residual,
                num_workers = 4
            )
            
            # Adds the dataloader to the dictionary
            self.partitions_info_dict[dataset_type]['dataloader'] = dataloader
        
        # If the datasets are empty, cannot train
        _is_dataset_created = self._check_create_dataset()
        
        return _is_dataset_created

    
    def _residual_compute_and_decide(self) -> bool:
        """
        Calculates the residual and choses to use it or not.
        
        Returns:
            drop_residual (bool): True to discard the residual.
        """

        # Calculates the residual
        residual = len(self.partitions_info_dict["training"]['files']) % self.configuration['hyperparameters']['batch_size']
        print("Residual for Batch training =", residual)
        
        
        # If the residual is too small, it is discarded
        if residual < (self.configuration['hyperparameters']['batch_size']/2):
            print("Residual discarded")
            drop_residual = True
        
        # If the residual is ok, it is used
        else:
            print("Residual not discarded")
            drop_residual = False
        
        return drop_residual


    def _shuffle_dataset(self, dataset, dataset_type):
        """
        Creates the shuffled dataset.
        It's a subdataset. The only difference with the base dataset is the image order.
        
        Args:
            _dataset (Custom2DDataset): The dataset to shuffle.
        
        Returns:
            _shuffled_dataset (Custom2DDataset): The shuffled dataset.
        """
                
        if dataset_type == 'training': # For the training
            do_shuffle = True # Lets the dataloader shuffle the images
        
        
        else: # For the other datasets
            do_shuffle = False # Doesn't let the dataloader shuffle the images
            
            # Shuffles the images manually to keep a track on the indexes
            indexes_list = list(range(len(dataset)))    # Gets the list of indexes
            random.shuffle(indexes_list)                # Shuffles it
            dataset = Subset(dataset, indexes_list)     # Creates a shuffled dataset


        return dataset, do_shuffle


    def _check_create_dataset(self):
        """
        Checks if the datasets are created.
        
        Returns:
            _is_dataset_created (bool): If the datasets are created or not.
        """

        # Checks if the training dataset is empty or not
        if self.partitions_info_dict['training']['dataloader'] is None:
            print(colored(
                f"Non-fatal Error: training was skipped for the Test Subject "
                f"{self.fold_info.testing_subject} and Subject {self.fold_info.validation_subject}. "
                f"There were no files in the training dataset.\n",
                'yellow'
            ))
            _is_dataset_created = False
        # Checks if the validation dataset is empty or not
        elif (self.is_cv_loop and 
              not self.partitions_info_dict['validation']['dataloader']):
            print(colored(
                f"Non-fatal Error: training was skipped for the Test Subject "
                f"{self.testing_subject} and Subject {self.validation_subject}. "
                f"There was no file in the validation dataset.\n",
                'yellow'
            ))
            _is_dataset_created = False
        # If both are not empty
        else:
            _is_dataset_created = True
            
        return _is_dataset_created
    
    
    def train_model(self):
        """
        Trains the model, assuming the given dataset is valid.
        """  
        
        
        # load model too
        
        # if self.checkpoint_epoch != 0 and \
        #    self.checkpoint_epoch == self.number_of_epochs:
        #     print(colored("Maximum number of epochs reached from checkpoint.", 'yellow'))
        #     return
        
        # Fits the model     
        fold_timer = PrecisionTimer()
        self.train()
        self.time_elapsed = fold_timer.get_elapsed_time() 


    def process_one_epoch(self, epoch_index, partition):       
           
        # Defines the data loader
        data_loader = self.get_dataloader(partition)
        
        # Initializations
        running_loss = 0.0
        running_corrects = 0
        
        self.all_labels = []
        self.all_predictions = []

        # Source: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#per-epoch-activity
        if partition == 'train':
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
        elif partition == "validation":
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()
            
        # Iterates over data
        for inputs, labels, _ in data_loader:
            
            # Runs the fit loop
            loss_update, corrects_update = self.process_batch(
                partition, inputs, labels,
                self.use_mixed_precision
            )
            
            running_loss += loss_update
            running_corrects += corrects_update


        # Calculates the loss and accuracy for the epoch and adds them to the history
        epoch_loss, epoch_accuracy = self.calculate_metrics_for_epoch(data_loader,
                                                                      partition,
                                                                      running_loss,
                                                                      running_corrects)

        self.loss_hist[partition][epoch_index] = epoch_loss
        self.accuracy_hist[partition][epoch_index] = epoch_accuracy
        
        if partition == 'validation' or not self.is_cv_loop:
            save_history_to_csv(self.history,
                                Path(self.configuration['output_path']),
                                self.test_fold_name,
                                self.validation_fold_name,
                                self.configuration["architecture_name"],
                                self.is_cv_loop)

        
        
        # Saves the best model
        if partition == 'validation':
            
            # print epoch results
            print(f' loss: {self.loss_hist["train"][epoch_index]:.4f} |'
                  f'val_loss: {epoch_loss:.4f} |'
                  f'accuracy: {self.accuracy_hist["train"][epoch_index]:.4f} |' f'val_accuracy: {epoch_accuracy:.4f}')

            is_best = False
            if epoch_loss < self.best_valid_loss:
                # self.save_best_model(epoch_index)
                is_best = True
                self.best_valid_loss = epoch_loss

            self.early_stopping(epoch_loss)
            self.counter_early_stopping = self.early_stopping.get_counter()
            self.do_early_stop = self.early_stopping.do_early_stop
            
            self.save_model(epoch_index, epoch_loss,
                            epoch_accuracy, is_best)
        
        elif not self.is_cv_loop:
            # print epoch results
            print(f' loss: {self.loss_hist["train"][epoch_index]:.4f} |'
                  f'accuracy: {self.accuracy_hist["train"][epoch_index]:.4f} |')

            self.save_model(epoch_index, epoch_loss,
                            epoch_accuracy, False)
            
        return 


    def train(self):
        """
        Fits the model.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=f'runs/nachosv2_{timestamp}')
        
        # TODO: get history if interruped by reading file
        
        checkpoint = self.load_checkpoint()
        # verify is less than the expected 
        #TODO: use values of checkpoint to load into model
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss_hist = checkpoint["loss_hist"]
            self.accuracy_hist = checkpoint["accuracy_hist"]
            self.best_valid_loss = checkpoint["best_val_loss"]
            self.history = checkpoint["history"] 

            self.counter_early_stopping = checkpoint["counter_early_stopping"]
            
            self.early_stopping = EarlyStopping(patience=self.hyperparameters['patience'],
                                           verbose=True,
                                           counter=self.counter_early_stopping,
                                           best_val_loss=self.best_valid_loss)        
        else:        
            self.loss_hist = {"train": [0.0] * self.number_of_epochs,
                              "validation": [0.0] * self.number_of_epochs}
            self.accuracy_hist = {"train": [0.0] * self.number_of_epochs,
                                  "validation": [0.0] * self.number_of_epochs}
            self.best_valid_loss = math.inf
            self.early_stopping = EarlyStopping(patience=self.hyperparameters['patience'],
                                           verbose=True,
                                           counter=self.counter_early_stopping,
                                           best_val_loss=self.best_valid_loss)    

        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # For each epoch
        for epoch in range(self.start_epoch, self.number_of_epochs):
            
            # Prints the current epoch
            print('-' * 60)
            print(f'Epoch {epoch + 1}/{self.number_of_epochs}')
            
            # Defines the list of partitions
            # AHPO/CV: ["train", "validation"]
            # Cross-testing: ["train"]
            partitions_list = self.get_partitions()
            
            for partition in partitions_list:
                self.process_one_epoch(epoch, partition)

            # ReduceLROnPlateau requires the validation loss
            self.scheduler.step()
            
            if self.is_cv_loop and self.do_early_stop:
                print("Early stopping")
                break
     

    def get_partitions(self) -> List[str]:
        """
        Get a list of partitions based on loop context.

        If the loop is not cross-testing loop, 'validation' is added to the partitions.

        Returns:
            list[str]: A list containing the partitions 'train' and optionally 'validation'.
        """
    
        partitions = ['train']
        if self.is_cv_loop:
            partitions.append('validation')
        
        return partitions


    def get_dataloader(self, partition: str) -> DataLoader:
        """
        Fetches the DataLoader for a given partition.

        Args:
            partition (str): The partition type ("train" or "validation").

        Returns:
            DataLoader: The DataLoader for the specified partition.

        Raises:
            ValueError: If the specified partition is not supported.
        """
        
        if partition == 'train': # Training phase
            data_loader = self.partitions_info_dict['training']['dataloader'] # Loads the training set
        elif partition == 'validation': # Validation phase
            data_loader = self.partitions_info_dict['validation']['dataloader'] # Loads the validation set
        elif partition == 'test': # Validation phase
            data_loader = self.partitions_info_dict['test']['dataloader'] # Loads the validation set
        else:
            raise ValueError(f"Partition {partition} is not valid.")
        return data_loader
    
    def process_batch_mixed_precision(self, partition, inputs, labels):
        # Uses mixed precision to use less memory
        with autocast(enabled=True, dtype=torch.float16,
                        cache_enabled=True):
            # Make predictions for this batch
            outputs = self.model(inputs)
            
            # Compute the loss
            loss = self.loss_function(outputs, labels)
        
            # Backward pass and optimize only if in training phase
            if partition == 'train':
                # scale the loss to avoid underflow
                # which can occur when using mixed precision float16
                self.scaler.scale(loss).backward()
                
                # Gradient clipping to avoid exploding gradients
                # it is required to unscale to float32
                # float16 might be too small to update the weights
                # scaler.unscale_(self.optimizer)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                #                             max_norm=1.0)
                
                # because of mixed precision, optimizer.step()
                # is replaced by
                self.scaler.step(self.optimizer)
                # Updates the scale for next iteration. 
                self.scaler.update()
        
        return outputs, loss

    
    def process_batch_standard(self, partition, inputs, labels):
        # forward + backward + optimize
        # Make predictions for this batch
        outputs = self.model(inputs)
        # Compute the loss and its gradients
        loss = self.loss_function(outputs, labels)
        
        if partition == "train":
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
        
        return outputs, loss

    def process_batch(self,
                      partition: str,
                      inputs,
                      labels,
                      use_mixed_precision: bool = False):
        """
        Runs the fit loop for the training or evaluation phase.

        Args:
            partition (str): The phase of the model ('train' or 'eval').
            inputs (Tensor): The input data for the current batch.
            labels (Tensor): The true labels corresponding to the inputs.
            
            running_loss (float): The cumulative loss for the current phase.
            running_corrects (int): The cumulative number of correct predictions for the current phase.

        Returns:
            running_loss (float): The updated cumulative loss after processing the batch.
            running_corrects (int): The updated cumulative number of correct predictions.
        """

        if partition not in ['train', 'validation']:
            raise ValueError(f"Unknown partition '{partition}'. Expected 'train' or 'validation'.")

        # https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.set_grad_enabled.html
        # sets gradient calculation on or off.
        # On: training
        # Off: validation, test
        with torch.set_grad_enabled(partition == 'train'):
            # Sends the inputs and labels to the execution device
            inputs = inputs.to(self.execution_device)
            labels = labels.to(self.execution_device)
            
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            
            if use_mixed_precision:
                outputs, loss = self.process_batch_mixed_precision(partition, inputs, labels)
            else:
                outputs, loss = self.process_batch_standard(partition, inputs, labels)
            
            # Converts the outputs to predictions 
            # max returns a tuple of two output tensors max, max_indices
            _, predictions = torch.max(outputs, dim=1)
                
            # Saves informations to calculates the recall
            # if 'recall' in self.metrics_dictionary and partition == 'validation':
            #     self.all_labels.append(labels)
            #     self.all_predictions.append(predictions)
                
            # Statistics
            batch_loss = loss.item() * inputs.size(0)
            batch_corrects = torch.sum(predictions == labels.data).item()

            return batch_loss, batch_corrects


    def calculate_metrics_for_epoch(self,
                                    data_loader: DataLoader,
                                    partition: str,
                                    running_loss: float,
                                    running_corrects: int):
        """
        Calculates the loss and accuracy for the epoch and adds them to the history.

        Args:
            _data_loader (DataLoader): The data loader providing the dataset for the current phase.
            phase (str): The phase of the model ('train' or 'eval').
            running_loss (float): The cumulative loss for the current epoch.
            running_corrects (int): The cumulative number of correct predictions for the current epoch.

        Returns:
            epoch_loss (float): The average loss for the epoch.
            epoch_accuracy (float): The accuracy for the epoch.
        """

        # Calculates the loss and accuracy of the current epoch 
        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_accuracy = float(running_corrects) / len(data_loader.dataset)
        
        # Saves the loss and accuracy of the current epoch into the history
        self.history[f'{partition}_loss'].append(epoch_loss)
        self.history[f'{partition}_accuracy'].append(epoch_accuracy)
        
        # if self.metrics_dictionary['recall'] and partition == 'validation':
        #     epoch_class_recall(self.fold_info.current_configuration['class_names'], epoch, self.all_labels, self.all_predictions, self.history)
        
        return epoch_loss, epoch_accuracy


    def save_model(self, epoch_index:int,
                   epoch_loss:float, epoch_accuracy:float,
                   is_best: bool = False):
        """
        Save model and other data.
        
        Args:
            best_model_path (str): The path where to save the best model.
            epoch (int): The current epoch.
            epoch_loss (float): The loss of the epoch.
            epoch_accuracy (float): The accuracy of the epoch.
            best_accuracy (float): The best accuracy obtained in the training.
        """       
        # If the directory does not exist, creates it
        if not self.checkpoint_folder_path.exists():
            self.checkpoint_folder_path.mkdir(mode=0o777, parents=True, exist_ok=True)
        
        # If the epoch is within the frequency steps, saves it
        checkpoint_frequency = self.configuration['checkpoint_epoch_frequency']
        is_frequency_checkpoint = ( (epoch_index+1) % checkpoint_frequency == 0)
        if is_frequency_checkpoint or is_best:

            l_path_to_save = []
            if is_frequency_checkpoint:
                checkpoint_file_path = self.checkpoint_folder_path / f"{self.prefix_name}_{epoch_index + 1}.pth"
                l_path_to_save.append(checkpoint_file_path)
            if is_best:            
                best_checkpoint_file_path = self.checkpoint_folder_path / f"{self.prefix_name}_{epoch_index + 1}_best.pth"
                l_path_to_save.append(best_checkpoint_file_path)
            
            for path in l_path_to_save:
                torch.save({"model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": epoch_index,
                            "counter_early_stopping": self.counter_early_stopping,
                            "best_val_loss": self.best_valid_loss,
                            "accuracy_hist": self.accuracy_hist,
                            "loss_hist": self.loss_hist,
                            "history": self.history
                           }, path)
            
                print(colored(f"Saved a checkpoint for epoch {epoch_index + 1}/{self.number_of_epochs} at {path}.", 'cyan'))
            
            # Keeps only the previous checkpoint for the most recent training fold, to save memory.           
            # Delete previous checkpoint
            if is_frequency_checkpoint:
                if self.prev_checkpoint_file_path and \
                   self.prev_checkpoint_file_path.exists():
                    os.remove(self.prev_checkpoint_file_path)
        
                self.prev_checkpoint_file_path = checkpoint_file_path
            if is_best:
                if self.prev_best_checkpoint_file_path and \
                   self.prev_best_checkpoint_file_path.exists():
                    os.remove(self.prev_best_checkpoint_file_path)

                self.prev_best_checkpoint_file_path = best_checkpoint_file_path

    # def save_best_model(self, epoch_index:int):
    #     """
    #     Saves the best model.
        
    #     Args:
    #         best_model_path (str): The path where to save the best model.
    #         epoch (int): The current epoch.
    #         epoch_loss (float): The loss of the epoch.
    #         epoch_accuracy (float): The accuracy of the epoch.
    #     """
                   
    #     # If the directory does not exist, creates it
    #     if not self.checkpoint_folder_path.exists():
    #         self.checkpoint_folder_path.mkdir(mode=0o777, parents=True, exist_ok=True)
    
    #     # Saves the checkpoint to file: Name_Current-epoch.pth
    #     new_save_path = self.checkpoint_folder_path / f"{self.prefix_name}_{epoch_index + 1}_best.pth"
        
    #     torch.save({"model_state_dict": self.model.state_dict(),
    #                 "optimizer_state_dict": self.optimizer.state_dict()},
    #                 new_save_path)
        
    #     print(colored(f"Saved a checkpoint for best model at epoch {epoch_index + 1}/{self.number_of_epochs} at {new_save_path}.", 'cyan'))
        
    #     # Keeps only the previous checkpoint for the most recent training fold, to save memory.
        
    #     # Delete previous checkpoint
    #     if self.prev_best_checkpoint_file_path and self.prev_best_checkpoint_file_path.exists():
    #         os.remove(self.prev_best_checkpoint_file_path)

    #     self.prev_best_checkpoint_file_path = new_save_path
       
    #     return 


    def get_checkpoint_info(self,
                            list_paths: List[Path]):
        """
        """
        last_checkpoint_epoch = 0
        last_checkpoint_path = None
        best_checkpoint_path = None
        
        for path in list_paths:
            # epochs in filename start at 1
            epoch_index = path.stem.split("_")[-1]
            if epoch_index == "best":
                best_checkpoint_path = path
                continue
            else:
                epoch_index = int(epoch_index) - 1
            if epoch_index >= last_checkpoint_epoch:
                last_checkpoint_epoch = epoch_index
                last_checkpoint_path = path

        return last_checkpoint_path, last_checkpoint_epoch, best_checkpoint_path


    def load_checkpoint(self):
        """
        """
        checkpoint = None
        # get list of checkpoints
        checkpoint_list = list(self.checkpoint_folder_path.glob(f"*{ self.prefix_name}*.pth"))
        
        # TODO: regex to obtain specific epoch
        last_checkpoint_path, last_checkpoint_epoch, best_checkpoint_path = self.get_checkpoint_info(checkpoint_list)
        
        if last_checkpoint_path is not None:
            self.prev_checkpoint_path = last_checkpoint_path
            # the checkpoint_epoch is finished
            # therefore the start should be the next one
            self.start_epoch = last_checkpoint_epoch + 1
            # retrieve specific checkpoint file
            checkpoint = torch.load(last_checkpoint_path,
                                    weights_only=True)

        if best_checkpoint_path is not None:
            self.prev_best_checkpoint_file_path = best_checkpoint_path
            
        return checkpoint

    
    def process_results(self):
        """
        Outputs the training results in files.
        """
        
        # best_model_path = "results/temp/temp_best_model.pth"
        
        # Loads the best model weights
        if self.is_cv_loop:
            checkpoint = torch.load(self.prev_best_checkpoint_file_path,
                                    weights_only=True)
        else:
            checkpoint = torch.load(self.prev_checkpoint_file_path,
                                    weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Sets the model to evaluation mode
        self.model.eval()
        
        # self.model.load_state_dict(torch.load(best_model_path))
                
        print(colored(f"Finished training for test subject {self.test_fold_name}"
                      f"and validation subject {self.validation_fold_name}.", 'green'))
        
        path_output_results = Path(self.configuration['output_path']) / 'training_results'
        
        predict_and_save_results(
            execution_device=self.execution_device,
            output_path=path_output_results, 
            test_fold_name=self.test_fold_name, 
            validation_fold_name=self.validation_fold_name, 
            model=self.model, 
            history=self.history, 
            time_elapsed=self.time_elapsed, 
            partitions_info_dict=self.partitions_info_dict, 
            class_names=self.configuration['class_names'],
            job_name=self.configuration['job_name'],
            architecture_name=self.configuration['architecture_name'],
            is_cv_loop=self.is_cv_loop,
            rank=self.mpi_rank
        )
        