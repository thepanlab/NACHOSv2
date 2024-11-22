import random
from typing import Optional, Callable, Union, Tuple, List

from collections import OrderedDict

from termcolor import colored
import pandas as pd

# import torch
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

from nachosv2.training.training_processing.custom_2D_dataset import Dataset2D
from nachosv2.training.training_processing.custom_3D_dataset import Custom3DDataset
from nachosv2.training.training_processing.training_fold_informations import TrainingFoldInformations
from nachosv2.data_processing.create_history import create_history
from nachosv2.data_processing.normalizer import normalizer
from nachosv2.image_processing.image_crop import create_crop_box
from nachosv2.image_processing.image_parser import *
from nachosv2.checkpoint_processing.checkpointer import *
from nachosv2.checkpoint_processing.delete_log import *
from nachosv2.checkpoint_processing.read_log import read_item_list_in_log
from nachosv2.checkpoint_processing.load_save_metadata_checkpoint import save_metadata_checkpoint
from nachosv2.model_processing.save_model import save_model
from nachosv2.modules.timer.precision_timer import PrecisionTimer
from nachosv2.output_processing.result_outputter import output_results
from nachosv2.results_processing.class_recall.epoch_recall import epoch_class_recall
from nachosv2.model_processing.initialize_model_weights import initialize_model_weights        
from nachosv2.model_processing.create_model import create_training_model
from nachosv2.model_processing.get_metrics_dictionary import get_metrics_dictionary
from nachosv2.modules.optimizer.optimizer_creator import create_optimizer
from typing import Union, List



class TrainingFold():
    def __init__(
        self,
        execution_device: str,
        rotation_index: int,
        configuration: dict,
        test_subject: str,
        validation_subject: str,
        list_training_subjects: list,
        df_metadata: pd.DataFrame,
        number_of_epochs: int,
        do_normalize_2d: bool = False,
        mpi_rank: int = None,
        is_outer_loop: bool = False,
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
            test_subject,           # The test subject name
            validation_subject,     # The validation subject name
            list_training_subjects, # A list of fold partitions
            df_metadata,            # The data dictionary
            number_of_epochs,       # The number of epochs
            do_normalize_2d,        #
            mpi_rank,               # An optional value of some MPI rank
            is_outer_loop,          # If this is of the outer loop
            is_3d,                  #
            is_verbose_on           # If the verbose mode is activated
        )
        
        self.rotation_index = rotation_index
        self.configuration = configuration
        self.hyperparameters = configuration['hyperparameters']
        
        self.test_subject = test_subject
        self.validation_subject = validation_subject
        self.list_training_subjects = list_training_subjects
        
        self.df_metadata = df_metadata
        self.number_of_epochs = number_of_epochs
        self.metrics_dictionary = get_metrics_dictionary(self.configuration['metrics'])
        
        self.partitions_info_dict = OrderedDict( # Dictionary of dictionaries
            [('training', {'files': [], 'labels': [], 'dataloader': None}),
             ('validation', {'files': [], 'labels': [], 'dataloader': None}),
             ('test', {'files': [], 'labels': [], 'dataloader': None}),
            ]
            )
        
        self.list_callbacks = None
        self.loss_function = nn.CrossEntropyLoss()
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        self.execution_device = execution_device
        self.history = create_history(self.fold_info.metrics_dictionary)
        self.time_elapsed = None
        
        self.partitions_info_dict = OrderedDict( # Dictionary of dictionaries
            [('training', {'files': [], 'labels': [], 'dataloader': None}),
             ('validation', {'files': [], 'labels': [], 'dataloader': None}),
             ('test', {'files': [], 'labels': [], 'dataloader': None}),
            ]
            )
        
        self.checkpoint_epoch = 0
        self.mpi_rank = mpi_rank
        self.is_outer_loop = is_outer_loop
        self.is_3d = is_3d
        self.do_normalize_2d = do_normalize_2d
        
        self.crop_box = create_crop_box(
            self.hyperparameters['cropping_position'][0], # The height of the offset
            self.hyperparameters['cropping_position'][1], # The width of the offset
            self.configuration['target_height'],  # The height wanted
            self.configuration['target_width'],   # The width wanted
            is_verbose_on
        )
        
        # If MPI, specifies the job name by task
        if self.mpi_rank:
            job_name = configuration['job_name'] # Only to put in the new job name
            
            # Checks if inner or outer loop
            if is_outer_loop: # Outer loop => no validation
                new_job_name = f"{job_name}_test_{test_subject}" 
            else: # Inner loop => validation
                new_job_name = f"{job_name}_test_{test_subject}_val_{validation_subject}"

            # Updates the job name
            self.configuration['job_name'] = new_job_name
            
        if is_verbose_on:  # If the verbose mode is activated
            print(colored("Fold of training informations created.", 'cyan'))


    def create_callbacks(self):
        """
        Creates the training callbacks. 
        This includes early stopping and checkpoints.
        """
        
        # Gets the job name to create the checkpoint prefix
        if self.is_outer_loop:          
            self.checkpoint_prefix = f"{self.configuration['job_name']}_test_{ self.fold_info.test_subject}" + \
                                     f"_config_{self.configuration['selected_model_name']}"
            
        else:
            self.checkpoint_prefix = f"{self.configuration['job_name']}_test_{ self.fold_info.test_subject}" + \
                                     f"_val_{self.validation_subject}_config_{self.configuration['selected_model_name']}"
        
        # Creates the path where to save the checkpoint
        save_path = os.path.join(self.configuration['output_path'], 'checkpoints')
        
        # Creates the checkpoint
        checkpointer = Checkpointer(
            self.number_of_epochs,
            self.configuration['k_epoch_checkpoint_frequency'],
            self.checkpoint_prefix,
            self.mpi_rank,
            save_path
        )


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
            if partition == 'testing':
                value_dict['files'] = \
                    self.df_metadata[self.df_metadata["subject"] == self.test_subject]["absolute_filepath"].tolist()
                value_dict['labels'] = \
                    self.df_metadata[self.df_metadata["subject"] == self.test_subject]["label"].tolist()
            elif partition == 'validation':
                value_dict['files'] = \
                    self.df_metadata[self.df_metadata["subject"] == self.validation_subject]["absolute_filepath"].tolist()
                value_dict['labels'] = \
                    self.df_metadata[self.df_metadata["subject"] == self.validation_subject]["label"].tolist()
            else:  # training
                value_dict['files'] = \
                    self.df_metadata[self.df_metadata["subject"].isin(self.list_training_subjects)]["absolute_filepath"].tolist()
                value_dict['labels'] = \
                    self.df_metadata[self.df_metadata["subject"].isin(self.list_training_subjects)]["label"].tolist()  


    def run_preparations(self):
        """
        Runs all of the steps in order to create the basic training information.
        """
        
        self.get_dataset_info()
        self.create_model()
        self.optimizer = create_optimizer(self.model,
                                          self.configuration['hyperparameters'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', factor = 0.1, patience = self.configuration['hyperparameters']['patience'], verbose = True)
        self.create_callbacks()


    def create_model(self):
        """
        Creates the initial model for training and initializes its weights.
        """
        
        # Creates the model
        self.model = create_training_model(self.configuration)
        
        # Initializes the weights
        initialize_model_weights(self.model)


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
        #     self.fold_i?state()
            
        # self.fold_info.run_all_steps()
        # Replace for all substeps
                
        self.get_dataset_info()
        self.create_model()
        self.optimizer = create_optimizer(self.model,
                                          self.configuration['hyperparameters'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', factor = 0.1, patience = self.configuration['hyperparameters']['patience'], verbose = True)
        self.create_callbacks()
        self.save_state()
            
        # Creates the datasets and trains them (Datasets cannot be logged.)
        if self.create_dataset():
            self.train_model()
            self._output_results()


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
            new_key_dict={'fold_info': self.fold_info},
            use_lock=self.mpi_rank != None
        )


    def load_checkpoint(self):
        """
        Loads the latest checkpoint to start from.
        """
        
        checkpoint = get_most_recent_checkpoint(
            os.path.join(self.fold_info.configuration['output_path'], 'checkpoints'),
            self.fold_info.checkpoint_prefix,
            self.fold_info.model
        )
        
        if checkpoint is not None:
            print(colored(f"Loaded most recent checkpoint of epoch: {checkpoint[1]}.", 'cyan'))
            self.fold_info.model.model = checkpoint[0]
            self.checkpoint_epoch = checkpoint[1]
        # If no checkpoint
        else:
            # Creates the model
            self.fold_info.create_model()
    
    
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
        for images, _ in dataloader:
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
                    dictionary_partition = self.partitions_info_dict[dataset_type],               # The data dictionary
                    number_channels = self.hyperparameters['number_channels'],                     # The number of channels in an image
                    do_cropping = self.hyperparameters['do_cropping'],                  # Whether to crop the image
                    crop_box = self.crop_box,                                        # The dimensions after cropping
                    transform=transform
                )

            # TODO verify the shuffle           
            # Shuffles the images
            dataset, do_shuffle = self._shuffle_dataset(dataset, dataset_type)

            # Apply normalization to the dataset

            # Creates the data loader
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
                f"Non-fatal Error: training was skipped for the Test Subject {self.fold_info.testing_subject} and Subject {self.fold_info.validation_subject}. " + 
                f"There were no files in the training dataset.\n",
                'yellow'
            ))
            _is_dataset_created = False
        
        
        # Checks if the validation dataset is empty or not
        elif (not self.is_outer_loop and self.partitions_info_dict['validation']['ds'] is None):
            print(colored(
                f"Non-fatal Error: training was skipped for the Test Subject {self.testing_subject} and Subject {self.validation_subject}. " + 
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
        
        if self.checkpoint_epoch != 0 and \
           self.checkpoint_epoch == self.fold_info.number_of_epochs:
            print(colored("Maximum number of epochs reached from checkpoint.", 'yellow'))
            return
        
        # Fits the model        
        fold_timer = PrecisionTimer()
        
        self.fit_model()
        
        self.time_elapsed = fold_timer.get_elapsed_time() 


    def fit_model(self):
        """
        Fits the model.
        """
        
        # Initializations
        best_accuracy = 0.0
        best_model_path = "results/temp/temp_best_model.pth"
        
        # Sends the model to the execution device
        self.model.to(self.execution_device)
        
        # Makes sure the temp directory exists
        if not os.path.exists("results/temp"):
            os.makedirs("results/temp")
        
        # Read epoch from metadata checkpoint
        
        # For each epoch
        for epoch in range(self.number_of_epochs):
            
            # Prints the current epoch
            print('-' * 60)
            print(f'Epoch {epoch + 1}/{self.fold_info.number_of_epochs}')
            
            # Defines the list of phases
            list_of_phases = self._get_list_of_phases()
            
            # Creates the gradscaler
            scaler = GradScaler()
            
            # For each phase
            for phase in list_of_phases:
                
                # Defines the data loader
                data_loader = self._get_phase_dataloader(phase)
                
                # Initializations
                running_loss = 0.0
                running_corrects = 0
                
                self.all_labels = []
                self.all_predictions = []


                # Iterates over data
                for inputs, labels, in data_loader:
                    
                    # Runs the fit loop
                    running_loss, running_corrects, scaler = self._fit_loop(phase, inputs, labels, running_loss, running_corrects, scaler)


                # Calculates the loss and accuracy for the epoch and adds them to the history
                epoch_loss, epoch_accuracy = self._calculating_epoch_metrics(data_loader, phase, running_loss, running_corrects, epoch)


                # Saves the best model
                if phase == 'validation':
                    best_accuracy = self._save_best_model(best_model_path, epoch, epoch_loss, epoch_accuracy, best_accuracy)

            self.scheduler.step(epoch_loss)

    
    def _get_list_of_phases(self):
        """
        Defines the list of phases.
        
        Returns:
            list_of_phases (list of str): The list of phases.
        """
        
        # Initializes the list of phases
        list_of_phases = ['train']
        
        
        # If inner loop
        if not self.is_outer_loop:
            list_of_phases.append('validation') # Adds the validation phase
        
        
        return list_of_phases


    def _get_phase_dataloader(self, phase):
        """
        Defines the data loader for the fit depending of the phase.
        
        Args:
            phase (str): The training phase.
        
        Returns:
            data_loader (): The data loader.
        """
        
        if phase == 'train': # Training phase
            self.model.train()  # Sets model to training mode
            data_loader = self.partitions_info_dict['training']['dataloader'] # Loads the training set
            
        else: # Validation phase
            self.model.eval()   # Sets model to evaluate mode
            data_loader = self.partitions_info_dict['validation']['dataloader'] # Loads the validation set

        return data_loader
    

    def _fit_loop(self, phase, inputs, labels, running_loss, running_corrects, scaler):
        """
        Runs the fit loop for the training or evaluation phase.

        Args:
            phase (str): The phase of the model ('train' or 'eval').
            inputs (Tensor): The input data for the current batch.
            labels (Tensor): The true labels corresponding to the inputs.
            
            running_loss (float): The cumulative loss for the current phase.
            running_corrects (int): The cumulative number of correct predictions for the current phase.
            scaler (GradScaler): A gradient scaler for mixed precision training.

        Returns:
            running_loss (float): The updated cumulative loss after processing the batch.
            running_corrects (int): The updated cumulative number of correct predictions.
            scaler (GradScaler): The gradient scaler, potentially updated.
        """

        with torch.set_grad_enabled(phase == 'train'):
            # Sends the inputs and labels to the execution device
            inputs = inputs.to(self.execution_device)
            labels = labels.to(self.execution_device)
            
            # Zeros the parameter gradients
            self.optimizer.zero_grad()
            
            # Uses mixed precision to use less memory
            with autocast():
                # Passes the data into the model
                outputs = self.model(inputs)
                
                # Computes the loss
                loss = self.fold_info.loss_function(outputs, labels)
            
            # Converts the outputs to predictions
            _, predictions = torch.max(outputs, 1)

            # Backward pass and optimize only if in training phase
            if phase == 'train':
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    
                scaler.step(self.optimizer)
                scaler.update()
            
            
            # Saves informations to calculates the recall
            if self.fold_info.metrics_dictionary['recall'] and phase == 'validation':
                self.all_labels.append(labels)
                self.all_predictions.append(predictions)
                
                
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(predictions == labels.data)

        return running_loss, running_corrects, scaler


    def _calculating_epoch_metrics(self, _data_loader, phase, running_loss, running_corrects, epoch):
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
        epoch_loss = running_loss / len(_data_loader.dataset)
        epoch_accuracy = running_corrects.double() / len(_data_loader.dataset)
        
        
        # Saves the loss and accuracy of the current epoch into the history
        self.history[f'{phase}_loss'].append(epoch_loss)
        self.history[f'{phase}_accuracy'].append(epoch_accuracy.item())
        
        
        # Prints with 4 digits after the decimal point
        print(f'{phase} loss: {epoch_loss:.4f} | accuracy: {epoch_accuracy:.4f}')
        
        
        if self.fold_info.metrics_dictionary['recall'] and phase == 'validation':
            epoch_class_recall(self.fold_info.current_configuration['class_names'], epoch, self.all_labels, self.all_predictions, self.history)
        
        
        return epoch_loss, epoch_accuracy
    
    
    
    def _save_best_model(self, best_model_path, epoch, epoch_loss, epoch_accuracy, best_accuracy):
        """
        Saves the best model.
        
        Args:
            best_model_path (str): The path where to save the best model.
            epoch (int): The current epoch.
            epoch_loss (float): The loss of the epoch.
            epoch_accuracy (float): The accuracy of the epoch.
            best_accuracy (float): The best accuracy obtained in the training.
        """
        
        for callback in self.list_callbacks:
            
            if isinstance(callback, Checkpointer):
                callback.on_epoch_end(epoch, self.model)
                
            elif isinstance(callback, lr_scheduler.ReduceLROnPlateau):
                callback.step(epoch_loss)
        
        # Saves the best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            save_model(self.model, best_model_path)    
        
        return best_accuracy


    def _output_results(self):
        """
        Outputs the training results in files.
        """
        
        best_model_path = "results/temp/temp_best_model.pth"
        
        # Loads the best model weights
        self.model.load_state_dict(torch.load(best_model_path))
        
        
        print(colored(f"Finished training for testing subject {self.fold_info.testing_subject} and subject {self.fold_info.validation_subject}.", 'green'))
        
        output_results(
            execution_device = self.execution_device,
            output_path = os.path.join(self.fold_info.current_configuration['output_path'], 'training_results'), 
            testing_subject = self.fold_info.testing_subject, 
            validation_subject = self.fold_info.validation_subject, 
            trained_model = self.fold_info.model, 
            history = self.history, 
            time_elapsed = self.time_elapsed, 
            datasets = self.partitions_info_dict, 
            class_names = self.fold_info.current_configuration['class_names'],
            job_name = self.fold_info.current_configuration['job_name'],
            config_name = self.fold_info.current_configuration['selected_model_name'],
            loss_function = self.fold_info.loss_function,
            is_outer_loop = self.is_outer_loop,
            rank = self.fold_info.mpi_rank
        )
        