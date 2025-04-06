"""
Green: indications about where is the training
Cyan: verbose mode
Magenta: results
Yellow: warnings
Red: errors, fatal or not
"""
from pathlib import Path
import itertools
from typing import List, Dict, Union
import random
from termcolor import colored
import pandas as pd
import torch
from mpi4py import MPI
from nachosv2.data_processing.read_metadata_csv import read_metadata_csv
from nachosv2.data_processing.check_unique_subjects import check_unique_subjects
from nachosv2.setup.utils_training import is_image_3D
from nachosv2.training.training_processing.partitions import generate_dict_folds_for_partitions
from nachosv2.training.training_processing.training_fold import TrainingFold
from nachosv2.modules.timer.precision_timer import PrecisionTimer
from nachosv2.modules.timer.write_timing_file import write_timing_file
from nachosv2.output_processing.memory_leak_check import initiate_memory_leak_check, end_memory_leak_check
from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.get_config import get_config
from nachosv2.checkpoint_processing.load_save_metadata_checkpoint import write_log
from nachosv2.training.hpo.hpo import get_hp_configuration


def create_loop_indices(config_dict: dict,
                        is_cv_loop: bool) -> List[dict]:
    """
    Create a list of dictionaries containing loop indices for training, that is training, validation(if cross-validation loop), and testing.

    Args:
        test_fold_list (list): List of test folds.
        hpo_list (list of dict]): List of hyperparameter optimization configurations.
        validation_fold_list (list): List of validation folds.

    Returns:
        List[dict]: List of dictionaries with keys 'test', 'hpo', and 'validation' 
                    representing the loop indices.
    """
    
    validation_fold_list = get_fold_list("validation",
                                         is_cv_loop,
                                         config_dict)
    test_fold_list = get_fold_list("test",
                                   is_cv_loop,
                                   config_dict)

    if config_dict["use_hpo"]:
        random.seed(config_dict['seed_hpo'])

    hp_list = get_hp_configuration(config_dict)  
    
    list_loop_indices = []
    for t, h, v in itertools.product(test_fold_list,
                                     hp_list,
                                     validation_fold_list):
        if t != v:
            list_loop_indices.append({"test": t,
                                      "hp_configuration": h,
                                      "validation": v})
    
    return list_loop_indices


def perform_single_training(index: int,
                            n_combinations: int,
                            indices_loop_dict: dict,
                            is_cv_loop: bool,
                            df_metadata: pd.DataFrame,
                            execution_device: str,
                            config_dict: dict,
                            is_verbose_on: bool = False):
    """
    Executes a single training run for a given fold and hyperparameter configuration,
    handling cross-validation or cross-testing loop scenarios. This function sets up the 
    appropriate data partitions, configuration parameters, and performs logging before 
    and after model training.

    Parameters:
    ----------
    index : int
        Index of the current training iteration within the loop.
    n_combinations : int
        Total number of training combinations (used for progress display).
    indices_loop_dict : dict
        Dictionary containing fold and hyperparameter configuration details.
        Expected keys: "test", "validation", "hp_configuration".
    is_cv_loop : bool
        Indicates whether the function is being run as part of a cross-validation loop.
    df_metadata : pd.DataFrame
        Metadata DataFrame containing information related to training samples.
    execution_device : str
        Device identifier for computation (e.g., "cuda:0", "cpu").
    config_dict : dict
        Dictionary with full configuration for training, including:
        - fold list
        - output path
        - image processing options
        - other training settings.
    is_verbose_on : bool, optional
        Enables verbose output during training if True (default is False).

    Returns:
    -------
    None
        This function performs training and logging but does not return a value.
    """

    test_fold = indices_loop_dict["test"]
    validation_fold = indices_loop_dict["validation"]
    
    hpo_configuration = indices_loop_dict["hp_configuration"]
    hp_config_index = hpo_configuration["hp_config_index"]

    # Defines if the images are 2D or 3D based
    # on configuration image_size dimensions
    is_3d = is_image_3D(config_dict)
    
    do_normalize_2d = config_dict["do_normalize_2d"]   
    use_mixed_precision = config_dict["use_mixed_precision"]

    # cross-validation loop
    if is_cv_loop:
        print(colored(f'--- Training: {index + 1}/{n_combinations} ---',
                       'magenta'))
    # cross-testing loop
    else:
        print(colored(f'--- Training: {index + 1}/{n_combinations} ---',
                       'magenta'))

    print("Test fold:", test_fold)
    print("Hyperparameter configuration index:", hp_config_index)
    print("Validation fold:", validation_fold)

    partitions_dict = generate_dict_folds_for_partitions(
        validation_fold_name=validation_fold,
        is_cv_loop=is_cv_loop,
        fold_list=config_dict['fold_list'],
        test_fold_name=test_fold
        ) 

    write_log(
        config=config_dict,
        indices_dict=indices_loop_dict,
        training_index=index,
        is_training_finished=False,
        output_directory=config_dict['output_path'],
        is_cv_loop=is_cv_loop,
        rank=None,
    )

    training_folds_list = partitions_dict['training']

    # Creates and runs the training fold for this subject pair
    training_fold = TrainingFold(
        execution_device=execution_device,  # The name of the device that will be use
        training_index=index,  # The fold index within the loop
        configuration=config_dict,  # The training configuration
        indices_loop_dict=indices_loop_dict,  # The validation_subject name
        training_folds_list=training_folds_list,  # A list of fold partitions
        df_metadata=df_metadata,  # The data dictionary
        do_normalize_2d=do_normalize_2d,
        use_mixed_precision=use_mixed_precision,
        is_cv_loop=is_cv_loop,  # If this is of the outer loop. Default is false. (Optional)
        is_3d=is_3d,  # 
        is_verbose_on=is_verbose_on  # If the verbose mode is activated. Default is false. (Optional)
    )
    
    training_fold.run_all_steps()
    
    write_log(
        config=config_dict,
        indices_dict=indices_loop_dict,
        training_index=index,
        is_training_finished=True,
        output_directory=config_dict['output_path'],
        is_cv_loop=is_cv_loop,
        rank=None,
    )


def get_fold_list(partition: str,
                  is_cv_loop: bool,
                  config_dict: dict) -> List[Union[str, None]]:
    """
    Retrieves the list of folds to be used for a specified data partition 
    ("validation" or "test") based on the current loop type and configuration.

    Parameters:
    ----------
    partition : str
        The data partition to retrieve fold list for. Must be either "validation" or "test".
    is_cv_loop : bool
        Indicates if the current execution is part of a cross-validation loop.
    config_dict : dict
        Configuration dictionary containing fold information. Expected keys:
        - "validation_fold_list"
        - "test_fold_list"
        - "fold_list"

    Returns:
    -------
    List[Union[str, None]]
        A list of fold names to be used for the given partition. 
        - For validation in non-CV mode, returns [None].
        - For test with no specific test folds provided, returns all folds from `config_dict['fold_list']`.

    Raises:
    ------
    ValueError
        If an invalid partition name is provided.
    """
    
    # If the test_subjects list is empty, uses all subjects

    if partition not in ["validation", "test"]:
        raise ValueError(f"Invalid partition: {partition}")

    partition_map = {"validation": "validation_fold_list",
                     "test": "test_fold_list"}
    
    fold_list = config_dict.get(partition_map[partition])

    def normalize_to_list(fold_value):
        if isinstance(fold_value, str):
            return [fold_value]
        elif isinstance(fold_value, list):
            return fold_value

    if partition == "validation":
        if not is_cv_loop:
            if config_dict['validation_fold_list'] is not None:
                print(colored("For cross-testing, validation_fold_list is not used"), "yellow")
            return [None]
    else:
        if not fold_list:  # If fold_list is None or empty
            print(colored("Not fold provided for test fold. Using all folds in fold_list"), "yellow")            
            return normalize_to_list(fold_list)
    return normalize_to_list(fold_list)


def get_index_device(num_device_to_use: int,
                     rank: int,
                     enable_dummy_node: bool):
    """
    Determines the GPU index to be used by a process based on its rank in a distributed setup.

    Parameters:
    ----------
    num_device_to_use : int
        The number of GPUs intended for use during training.
    rank : int
        The rank of the current process (in a multi-process/distributed setting).
    enable_dummy_node : bool
        If True, assumes the presence of a dummy node (e.g., for coordination)
        and adjusts GPU indexing to skip over it.

    Returns:
    -------
    int
        The index of the GPU that should be assigned to this process.

    Notes:
    ------
    - When `enable_dummy_node` is True, the modulo operation is performed with (num_device_to_use + 1),
      effectively treating one rank (usually rank 0) as a dummy that doesn't run training.
    - GPU indexing starts after the dummy node, with remaining ranks mapped to GPUs in a round-robin manner.
    """
    num_gpus_available = torch.cuda.device_count()
    
    print(colored(f'Rank {rank}', 'cyan'))
    print("Num GPUs Available: ", num_gpus_available)
    print("Num GPUs to use: ", num_device_to_use)

    if enable_dummy_node:
        # In distributed setups, rank 0 is skipped for training.
        # Rank 0 is the manager process
        # We map ranks to GPU indices using modulo arithmetic, adjusted for the dummy.
        # mod number is (# gpus +1)
        # Example with 2 GPUs and dummy enabled:
        #     Rank:        0   1   2   3   4   5   6
        #     GPU index:   2   0   1   2   0   1   2
        # Assuming you are using 2 GPUs
        # Valida GPUs indices are 0 and 1    
        
        index_gpu = (rank-1) % (num_device_to_use+1)
        print("index_gpu =", index_gpu)
        
        # The GPU index 2 will do nothing
        # Convert invalid GPU index to -1
        if index_gpu == num_device_to_use:
            index_gpu = -1
        
    else:
        index_gpu = (rank-1) % num_device_to_use

    print("rank", rank, "index_gpu", index_gpu)
    return index_gpu


def determine_if_cv_loop(loop: str) -> bool:
    """
    Determines whether the given loop type corresponds to a cross-validation loop.

    Parameters:
    ----------
    loop : str
        The type of training loop. Must be either "cross-validation" or "cross-testing".

    Returns:
    -------
    bool
        True if the loop is "cross-validation", False if "cross-testing".

    Raises:
    -------
    ValueError
        If the provided loop type is not one of the supported values.
    """
    if loop not in ["cross-validation", "cross-testing"]:
        raise ValueError(f"Invalid loop type: {loop}")

    if loop == 'cross-testing':
        is_cv_loop = False
    else:
        is_cv_loop = True
        
    return is_cv_loop


def train_sequential(config_dict: dict,
                     execution_device_list: list,
                     is_verbose_on: bool,
                     loop: str):
    """
    Executes a sequential training loop for either cross-validation or cross-testing
    using a single computational device.

    Parameters:
    ----------
    config_dict : dict
        Configuration dictionary containing training settings such as:
        - metadata CSV path
        - fold definitions
        - output paths
        - training/test loop types, etc.
    execution_device_list : list
        A list containing the single execution device to be used (e.g., ["cuda:0"]).
        Only one device is allowed in sequential mode.
    is_verbose_on : bool
        If True, enables verbose output for logging and status updates.
    loop : str
        Specifies the loop type: "cross-validation" or "cross-testing".

    Raises:
    -------
    ValueError
        If more than one device is passed in `execution_device_list`.

    Returns:
    -------
    None
        The function executes training and writes timing results, but does not return a value.
    """

    # Ensure only one device is used for sequential training
    if len(execution_device_list) > 1:
        raise ValueError("For sequential training, only one device can be used.")

    # # Checks for memory leaks
    # memory_leak_check_enabled = False
    # # TODO: verify this memory check
    # if memory_leak_check_enabled:
    #     memory_snapshot = initiate_memory_leak_check()

    # Determine whether we are in a cross-validation or cross-testing loop
    is_cv_loop = determine_if_cv_loop(loop)

    # Load metadata from the CSV file
    path_csv_metadata = config_dict["path_metadata_csv"]
    df_metadata = read_metadata_csv(path_csv_metadata)

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

    if is_verbose_on:  # If the verbose mode is activated
        print(colored("Double-checks of test and validation uniqueness successfully done.", 'cyan'))

    # Create the list of fold combinations for training
    indices_loop_list = create_loop_indices(config_dict,
                                            is_cv_loop)
    n_combinations = len(indices_loop_list)

    # Start measuring elapsed training time
    training_timer = PrecisionTimer()

    # Iterate through each combination of folds and train sequentially
    for index, indices_loop_dict in enumerate(indices_loop_list):
        perform_single_training(index=index,
                                n_combinations=n_combinations,
                                indices_loop_dict=indices_loop_dict,
                                is_cv_loop=is_cv_loop,
                                df_metadata=df_metadata,
                                execution_device=execution_device_list[0],
                                config_dict=config_dict,
                                is_verbose_on=is_verbose_on)
        
    # Report elapsed training time
    elapsed_time_seconds = training_timer.get_elapsed_time()
    print(colored(f"\nElapsed time: {elapsed_time_seconds:.2f} seconds.", 'magenta'))

    # Save the timing results to a file in the appropriate output directory
    loop_folder = "CT" if not is_cv_loop else "CV"
    timing_directory_path = Path(config_dict["output_path"]) / loop_folder /"training_timings" # The directory's path where to put the timing file
    timing_directory_path.mkdir(mode=0o775, parents=True, exist_ok=True)
    
    write_timing_file(training_timer,
                      timing_directory_path,
                      config_dict,
                      is_verbose_on)


def train_parallel(config_dict: dict,
                   execution_device_list: list,
                   enable_dummy_process: bool,
                   is_verbose_on: bool,
                   loop: str):
    """
    Executes training in parallel using MPI across multiple processes and devices.
    
    This function distributes training tasks across available processes. Rank 0 coordinates 
    the distribution, while worker ranks (rank > 0) listen for tasks, perform training, and 
    report completion.

    Parameters:
    ----------
    config_dict : dict
        Configuration settings including metadata paths, folds, and training parameters.
    execution_device_list : list
        List of available devices (e.g., ["cuda:0", "cuda:1"]).
    enable_dummy_process : bool
        If True, enables a dummy process for coordination when it is necessary to have 
        same number of processes per node.
    is_verbose_on : bool
        Enables detailed logging and console output.
    loop : str
        Specifies whether the loop is "cross-validation" or "cross-testing".

    Raises:
    -------
    ValueError
        If no training configurations are provided or misconfiguration is detected.
    """

    num_device_to_use = len(execution_device_list)
    
    # Initialize MPI communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_proc = comm.Get_size()
    
    print("rank", rank, "n_proc", n_proc)
    
    if rank == 0:
        # Master process controls configuration and task distribution
        is_cv_loop = determine_if_cv_loop(loop)

        # Check for uniqueness in validation folds if doing cross-validation
        if is_cv_loop:  # Only if we are in the inner loop
            check_unique_subjects(config_dict["validation_fold_list"],
                                  "validation")

        if is_verbose_on:
            print(colored("Double-checks of test and validation uniqueness successfully done.", 'cyan'))

        # Create task list for training (each entry is a fold/config combo)
        indices_loop_list = create_loop_indices(config_dict,
                                                is_cv_loop)
        n_tasks = len(indices_loop_list)

        if n_tasks == 0:
            # If no configurations exist, inform all workers and exit
            for subrank in range(1, n_proc):
                comm.send(False, dest=subrank)
            raise ValueError(colored("No configurations given.", 'yellow'))

        # Load metadata from CSV
        path_csv_metadata = config_dict["path_metadata_csv"]
        df_metadata = read_metadata_csv(path_csv_metadata)
        
                
        # Start a training timer
        training_timer = PrecisionTimer()
        
        # Assign training tasks to workers as they become available
        for index, indices_loop_dict in enumerate(indices_loop_list):
            subrank = comm.recv(source=MPI.ANY_SOURCE)
            dict_to_send = { 
                "index": index,
                "n_combinations": n_tasks,
                "indices_loop_dict": indices_loop_dict,
                "is_cv_loop": is_cv_loop,
                "df_metadata": df_metadata,
                "config_dict": config_dict,
                "is_verbose_on": is_verbose_on
            }
            comm.send(dict_to_send, dest=subrank)

        # Notify all workers that tasks are complete
        for subrank in range(1, n_proc):
            print(colored(f"Rank 0 is terminating rank {subrank}, no tasks to give.", 'red'))
            comm.send(False, dest=subrank)
        
        # Stop the timer and log elapsed time
        elapsed_time_seconds = training_timer.get_elapsed_time()
        print(colored(f"\nElapsed time: {elapsed_time_seconds:.2f} seconds.", 'magenta'))

        # Save timing information
        loop_folder = "CT" if not is_cv_loop else "CV"
        timing_directory_path = Path(config_dict["output_path"]) / loop_folder /"training_timings" # The directory's path where to put the timing file
        timing_directory_path.mkdir(mode=0o775, parents=True, exist_ok=True)
        
        write_timing_file(training_timer,
                          timing_directory_path,
                          config_dict,
                          is_verbose_on)
    else: 
        # Worker process (rank > 0)

        # Determine which GPU/device this rank should use
        index_device = get_index_device(num_device_to_use,
                                        rank,
                                        enable_dummy_process)
        
        # Listen for the first task
        print("num_device_to_use:", num_device_to_use)
        print(f"rank: {rank}, index_gpu: {index_device}")
        print("enable_dummy_process:", enable_dummy_process)
        
        if index_device != -1:
            # Notify rank 0 that this process is ready to receive a task
            comm.send(rank, dest=0)

            print(colored(f'Rank {rank} is listening for process 0.', 'cyan'))
            task = comm.recv(source=0)
            
            # Process training tasks as long as they are being sent
            while task:
                perform_single_training(
                        index=task["index"],
                        n_combinations=task["n_combinations"],
                        indices_loop_dict=task["indices_loop_dict"],
                        is_cv_loop=task["is_cv_loop"],
                        df_metadata=task["df_metadata"],
                        execution_device=execution_device_list[index_device],  # Use the GPU assigned to this rank
                        config_dict=task["config_dict"],
                        is_verbose_on=task["is_verbose_on"])

                # Notify rank 0 that this process is ready for a new task
                comm.send(rank, dest=0)
                task = comm.recv(source=0)


            print(colored(f'Rank {rank} terminated. All jobs finished for this process.', 'yellow'))


def train():
    """
    Initializes and executes the training process for a machine learning model. 
    Determines whether to run the training in sequential or parallel mode based 
    on the MPI communicator size.

    The function:
    - Parses command-line arguments for configuration settings.
    - Loads a configuration file for training parameters.
    - Determines whether to enable parallelization based on the MPI environment.
    - Executes either a sequential or parallel version of the training pipeline 
      based on the number of available MPI processes.
    """
    
    args = parse_command_line_args()

    # Extract argument values
    # Load training configuration from file
    config_dict = get_config(args['file'])
     # List of devices to be used for training 
    execution_device_list = args['devices']
    # Enable verbose output
    is_verbose_on = args['verbose']
    # Enable dummy process for parallel training
    enable_dummy_process = args['enable_dummy_process']
    # Whether to loop through training multiple times
    loop = args["loop"]

    # Initialize MPI communication
    comm = MPI.COMM_WORLD
    # Enable parallelization if more than one MPI process
    if comm.Get_size() > 1:
        enable_parallelization = True
    else:
        enable_parallelization = False

    print("execution_device_list",
          execution_device_list)
    # Execute the appropriate training method
    if not enable_parallelization:
        # Run the sequential version of training
        train_sequential(config_dict,
                         execution_device_list,
                         is_verbose_on,
                         loop)
    else:
        # Run the parallel version of training using MPI
        train_parallel(config_dict,
                       execution_device_list,
                       enable_dummy_process,
                       is_verbose_on,
                       loop)


if __name__ == "__main__":
    train()
