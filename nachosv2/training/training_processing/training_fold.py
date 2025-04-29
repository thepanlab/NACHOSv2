import random
import os
from collections import OrderedDict
from typing import Optional, Callable, Union, Tuple, List
from pathlib import Path
import math
from termcolor import colored
import pandas as pd
from datetime import datetime
# from tqdm import tqdm
# import torch
import torch
# from torch.cuda.amp import autocast, GradScaler
from torch import autocast, GradScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from nachosv2.training.training_processing.custom_2D_dataset import Dataset2D
from nachosv2.image_processing.image_crop import create_crop_box
# from nachosv2.image_processing.image_parser import *
from nachosv2.checkpoint_processing.checkpointer import Checkpointer
from nachosv2.checkpoint_processing.delete_log import delete_log_file
from nachosv2.checkpoint_processing.read_log import read_item_list_in_log
from nachosv2.checkpoint_processing.load_save_metadata_checkpoint import save_metadata_checkpoint
# from nachosv2.model_processing.save_model import save_model
from nachosv2.modules.timer.precision_timer import PrecisionTimer
from nachosv2.output_processing.result_outputter import predict_and_save_results
from nachosv2.model_processing.initialize_model_weights import initialize_model_weights        
from nachosv2.model_processing.create_model import create_model
from nachosv2.model_processing.get_metrics_dictionary import get_metrics_dictionary
from nachosv2.modules.optimizer.optimizer_creator import create_optimizer
from nachosv2.modules.early_stopping.earlystopping import EarlyStopping
from nachosv2.output_processing.result_outputter import save_history_to_csv
from nachosv2.setup.utils_training import create_empty_history
from nachosv2.setup.utils_training import get_files_labels
from nachosv2.setup.utils_training import get_mean_stddev


class TrainingFold():
    def __init__(
        self,
        execution_device: str,
        training_index: int,
        configuration: dict,
        indices_loop_dict: dict,
        training_folds_list: List[str],
        df_metadata: pd.DataFrame,
        do_normalize_2d: bool = False,
        use_mixed_precision: bool = False,
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

        self.training_index = training_index
        self.configuration = configuration
        self.indices_loop_dict = indices_loop_dict
        self.test_fold = indices_loop_dict["test"]
        self.validation_fold = indices_loop_dict["validation"]
        self.hyperparameters = indices_loop_dict["hp_configuration"]
        self.hp_config_index = self.hyperparameters["hp_config_index"]

        if is_cv_loop:
            self.prefix_name = f"{configuration['job_name']}" + \
                               f"_test_{self.test_fold}" + \
                               f"_hp_{self.hp_config_index}" + \
                               f"_val_{self.validation_fold}"
        else:
            self.prefix_name = f"{configuration['job_name']}" + \
                               f"_test_{self.test_fold}"

        self.training_folds_list = training_folds_list
        self.df_metadata = df_metadata
        # hyperparameter
        self.number_of_epochs = self.hyperparameters["n_epochs"]
        self.metrics_dictionary = get_metrics_dictionary(self.configuration['metrics_list'])
        
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # The input is expected to contain the unnormalized logits for each class
        # TODO: define as binary cross entropy 
        # when 2 classes are used
        # and modify predictions, they dont need to be softmaxed
        #https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        #https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586
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
        self.is_3d = is_3d
        self.do_normalize_2d = do_normalize_2d
        self.do_shuffle_the_images = configuration["do_shuffle_the_images"]

        if self.hyperparameters["do_cropping"]:
            self.crop_box = create_crop_box(
                self.hyperparameters['cropping_position']["x"], # The height of the offset
                self.hyperparameters['cropping_position']["y"], # The width of the offset
                self.configuration['target_height'],  # The height wanted
                self.configuration['target_width'],   # The width wanted
                is_verbose_on
            )
        else:
            self.crop_box = None

        loop_folder = 'CV' if self.is_cv_loop else 'CT'
        self.checkpoint_folder_path = Path(self.configuration['output_path']) / loop_folder /'checkpoints'
        self.start_epoch = 0

        self.prev_checkpoint_file_path = None
        self.prev_best_checkpoint_file_path = None
        self.last_checkpoint_file_path = None

        # TODO: verify it doesnt contradict with prefix
        if is_cv_loop: 
            new_job_name = f"{configuration['job_name']}_test_{self.test_fold}"+ \
                            f"_val_{self.validation_fold}"
        else: 
            new_job_name = f"{configuration['job_name']}_test_{self.test_fold}" 

            self.configuration['job_name'] = new_job_name


    def get_dataset_info(self):
        """
        Gets the basic data to create the datasets from.
        This includes the testing and validation_subjects, file paths,
        label indexes, and labels.
        """

        l_columns = ["absolute_filepath", "label"]

        # verify values in l_columns are in df_metadata.columns
        if not all(val_col in self.df_metadata.columns.to_list() for val_col in l_columns):
            raise ValueError(f"The columns {l_columns} must be in csv_metadata file.")

        for partition, value_dict in self.partitions_info_dict.items():
            value_dict['files'], value_dict['labels'] = get_files_labels(
                partition,
                self.df_metadata,
                self.test_fold,
                self.validation_fold,
                self.training_folds_list
            )


    def create_model(self):
        """
        Creates the initial model for training and initializes its weights.
        """

        # Creates the model
        self.model = create_model(self.configuration,
                                  self.hyperparameters)
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
        self.get_dataset_info()
        self.create_model()
        self.optimizer = create_optimizer(self.model,
                                          self.hyperparameters)

        # scheduler https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
      
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                              T_max=5,
                                                              eta_min=self.hyperparameters['learning_rate']/100,
                                                              last_epoch=-1,
                                                              verbose='deprecated')
        
        # self.save_state()
        # Creates the datasets and trains them (Datasets cannot be logged.)
        if self.create_dataset():
            fold_timer = PrecisionTimer()
            self.train()
            self.time_elapsed = fold_timer.get_elapsed_time() 
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


    def create_dataset(self) -> bool:
        """
        Creates the dataset needed to trains the model.
        It will map the image paths into their respective image and label pairs.
        TODO Complete any image-reading changes here for different file types.

        Returns:
            _is_dataset_created (bool): If the datasets are created or not.
        """

        # Gets the datasets for each phase
        for partition in self.partitions_info_dict:

            # If there is no files for the current dataset (= no validation = outer loop)
            if not self.partitions_info_dict[partition]['files']:
                continue  # Skips the rest of the loop

            drop_residual = False
            # If the current dataset is the training one
            if partition == "training":

                # Calculates the residual
                drop_residual = self._residual_compute_and_decide()

                transform = None

                if self.do_normalize_2d:
                    dataset_before_normalization = Dataset2D(
                        dictionary_partition=self.partitions_info_dict[partition],
                        number_channels=self.configuration['number_channels'],
                        image_size=(self.configuration['target_height'], self.configuration['target_width']),
                        do_cropping=self.hyperparameters['do_cropping'],
                        crop_box=self.crop_box,
                        transform=None
                    )

                    dataloader = DataLoader(
                        dataset=dataset_before_normalization,
                        batch_size=self.hyperparameters['batch_size'],
                        shuffle=False,
                        drop_last=drop_residual,
                        num_workers=0  # Default
                    )

                    mean, stddev = get_mean_stddev(
                        number_channels=self.configuration['number_channels'],
                        dataloader=dataloader)
                    transform = transforms.Normalize(mean=mean, std=stddev)

            # Creates the dataset
            if self.is_3d:
                dataset = Custom3DDataset(
                    self.configuration['data_input_directory'],   # The file path prefix = the path to the directory where the images are
                    self.partitions_info_dict[partition],               # The data dictionary
                )

            else:
                dataset = Dataset2D(
                    dictionary_partition=self.partitions_info_dict[partition], # The data dictionary
                    number_channels=self.configuration['number_channels'],      # The number of channels in an image
                    image_size=(self.configuration['target_height'], self.configuration['target_width']),
                    do_cropping=self.hyperparameters['do_cropping'],              # Whether to crop the image
                    crop_box=self.crop_box,                                       # The dimensions after cropping
                    transform=transform
                )

            # TODO verify the shuffle           
            # Shuffles the images
            do_shuffle = self.determine_shuffle_dataset(partition)

            # Apply normalization to the dataset

            # Creates dataloader
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.hyperparameters['batch_size'],
                shuffle=do_shuffle,
                drop_last=drop_residual,
                num_workers=4
            )

            # Adds the dataloader to the dictionary
            self.partitions_info_dict[partition]['dataloader'] = dataloader

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
        residual = len(self.partitions_info_dict["training"]['files']) % self.hyperparameters['batch_size']
        print("Residual for Batch training =", residual)

        # If the residual is too small, it is discarded
        if residual < (self.hyperparameters['batch_size']/2):
            print("Residual discarded")
            drop_residual = True

        # If the residual is ok, it is used
        else:
            print("Residual not discarded")
            drop_residual = False

        return drop_residual


    def determine_shuffle_dataset(self, partition):
        """
        Creates the shuffled dataset.
        It's a subdataset. The only difference with the base dataset is the image order.
        
        Args:
            _dataset (Custom2DDataset): The dataset to shuffle.
        
        Returns:
            _shuffled_dataset (Custom2DDataset): The shuffled dataset.
        """
        if not self.do_shuffle_the_images:
            return False
        return partition == 'training'


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


    def process_one_epoch(self, epoch_index, partition):

        # Defines the data loader
        data_loader = self.get_dataloader(partition)
        
        # Initializations
        running_loss = 0.0
        running_corrects = 0

        self.all_labels = []
        self.all_predictions = []

        # Source: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#per-epoch-activity
        if partition == 'training':
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
        elif partition == "validation":
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

        # Iterates over data
        for i, (inputs, labels, _ ) in enumerate(data_loader):
            print(f"Epoch: {epoch_index+1} of {self.number_of_epochs}. {partition} Progress: {i/len(data_loader)*100:.1f}%\r",
                  end="")
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

        # it saves history when self.is_cv_loop and for validation
        # or when not self.is_cv_loop, that is, cross-testing loop
        if partition == 'validation' or not self.is_cv_loop:
            self.history["learning_rate"].append(self.scheduler.get_last_lr()[0])
            save_history_to_csv(history=self.history,
                                output_path=Path(self.configuration['output_path']),
                                test_fold=self.test_fold,
                                hp_config_index=self.hp_config_index,
                                validation_fold=self.validation_fold,
                                is_cv_loop=self.is_cv_loop)

        # Saves the best model
        if partition == 'validation':
            # print epoch results
            print(f' loss: {self.loss_hist["training"][epoch_index]:.4f} |'
                  f'val_loss: {epoch_loss:.4f} | '
                  f'accuracy: {self.accuracy_hist["training"][epoch_index]:.4f} |' f'val_accuracy: {epoch_accuracy:.4f}')

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
            print(f' loss: {self.loss_hist["training"][epoch_index]:.4f} |'
                  f'accuracy: {self.accuracy_hist["training"][epoch_index]:.4f} |')
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
            self.loss_hist = {"training": [0.0] * self.number_of_epochs,
                              "validation": [0.0] * self.number_of_epochs}
            self.accuracy_hist = {"training": [0.0] * self.number_of_epochs,
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

                # Do a step                 
                if self.is_cv_loop: 
                    # In a cross-validation loop, step the scheduler only after validating                  
                    if partition == "validation":
                        self.scheduler.step()
                else:
                    # In a cross-testing loop, step the scheduler after training                  
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
        partitions = ['training']
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
        if partition == 'training': # Training phase
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
                      cache_enabled=True, device_type='cuda'):
            # Make predictions for this batch
            outputs = self.model(inputs)
            
            # Compute the loss
            loss = self.loss_function(outputs, labels)
        
            # Backward pass and optimize only if in training phase
            if partition == 'training':
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
        
        if partition == "training":
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
            partition (str): The phase of the model ('training' or 'eval').
            inputs (Tensor): The input data for the current batch.
            labels (Tensor): The true labels corresponding to the inputs.
            
            running_loss (float): The cumulative loss for the current phase.
            running_corrects (int): The cumulative number of correct predictions for the current phase.

        Returns:
            running_loss (float): The updated cumulative loss after processing the batch.
            running_corrects (int): The updated cumulative number of correct predictions.
        """

        if partition not in ['training', 'validation']:
            raise ValueError(f"Unknown partition '{partition}'. Expected 'training' or 'validation'.")

        # https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.set_grad_enabled.html
        # sets gradient calculation on or off.
        # On: training
        # Off: validation, test
        with torch.set_grad_enabled(partition == 'training'):
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
            phase (str): The phase of the model ('training' or 'eval').
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
            self.checkpoint_folder_path.mkdir(mode=0o775, parents=True, exist_ok=True)

        # If the epoch is within the frequency steps, saves it
        checkpoint_frequency = self.configuration['checkpoint_epoch_frequency']
        is_frequency_checkpoint = ( (epoch_index+1) % checkpoint_frequency == 0)
        # Determine if it is the last epoch
        is_last = ( epoch_index+1 == self.number_of_epochs)

        if is_frequency_checkpoint or is_best or is_last:

            # epoch index is 0-indexed
            # epoch number in file name is 1-indexed
            l_path_to_save = []
            if is_frequency_checkpoint:
                checkpoint_file_path = self.checkpoint_folder_path / f"{self.prefix_name}_epoch_{epoch_index + 1}.pth"
                l_path_to_save.append(checkpoint_file_path)
            if is_best:
                best_checkpoint_file_path = self.checkpoint_folder_path / f"{self.prefix_name}_epoch_{epoch_index + 1}_best.pth"
                l_path_to_save.append(best_checkpoint_file_path)
            if is_last:
                last_checkpoint_file_path = self.checkpoint_folder_path / f"{self.prefix_name}_epoch_{epoch_index + 1}_last.pth"
                l_path_to_save.append(last_checkpoint_file_path)

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
            if is_last:
                self.last_checkpoint_file_path = last_checkpoint_file_path


    def get_checkpoint_info(self,
                            list_paths: List[Path]):
        """
        """
        last_checkpoint_epoch = 0
        last_checkpoint_path = None
        best_checkpoint_path = None

        # epoch index is 0-indexed
        # epoch number in file name is 1-indexed

        for path in list_paths:
            # epochs in filename start at 1
            epoch_index = path.stem.split("_")[-1]
            if epoch_index == "best":
                best_checkpoint_path = path
            elif epoch_index == "last":
                last_checkpoint_epoch = int(path.stem.split("_")[-2]) - 1
                last_checkpoint_path = path
            elif int(epoch_index) >= last_checkpoint_epoch:
                last_checkpoint_epoch = int(epoch_index) - 1
                last_checkpoint_path = path

        return last_checkpoint_path, last_checkpoint_epoch, \
               best_checkpoint_path


    def load_checkpoint(self):
        """
        """
        checkpoint = None
        # get list of checkpoints
        checkpoint_list = list(self.checkpoint_folder_path.glob(f"*{self.prefix_name}*.pth"))
        
        # TODO: regex to obtain specific epoch
        last_checkpoint_path, last_checkpoint_epoch, best_checkpoint_path = self.get_checkpoint_info(checkpoint_list)

        if last_checkpoint_path is not None:
            self.prev_checkpoint_path = last_checkpoint_path
            # the checkpoint_epoch is finished
            # therefore the start should be the next one
            self.start_epoch = last_checkpoint_epoch + 1
            # retrieve specific checkpoint file
            checkpoint = torch.load(last_checkpoint_path,
                                    weights_only=True,
                                    map_location=self.execution_device)

        if best_checkpoint_path is not None:
            self.prev_best_checkpoint_file_path = best_checkpoint_path

        return checkpoint


    def process_results(self):
        """
        Outputs the training results in files.
        """
        
        # best_model_path = "results/temp/temp_best_model.pth"
        
        # Loads the best model weights
        checkpoint_path = (
            self.prev_best_checkpoint_file_path if self.is_cv_loop
            else self.last_checkpoint_file_path
        )

        print(colored(f"For predictions using checkpoint file: {checkpoint_path}", 'green'))
        checkpoint = torch.load(checkpoint_path,
                                weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Sets the model to evaluation mode
        self.model.eval()

        # self.model.load_state_dict(torch.load(best_model_path))
        if self.is_cv_loop:
            end_message = f"Finished training for test fold '{self.test_fold}'" + \
                          f" and validation fold '{self.validation_fold}'."
        else:
            end_message = f"Finished training for test fold '{self.test_fold}'"

        print(colored(end_message, 'green'))

        predict_and_save_results(
            execution_device=self.execution_device,
            output_path=Path(self.configuration['output_path']), 
            test_fold=self.test_fold,
            hp_config_index=self.hp_config_index,
            validation_fold=self.validation_fold, 
            model=self.model, 
            time_elapsed=self.time_elapsed, 
            partitions_info_dict=self.partitions_info_dict, 
            class_names=self.configuration['class_names'],
            is_cv_loop=self.is_cv_loop,
            enable_prediction_on_test=self.configuration.get('enable_prediction_on_test',False)
        )
