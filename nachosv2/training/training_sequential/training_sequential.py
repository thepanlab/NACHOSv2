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
# from nachosv2.training.training_sequential.execute_training import execute_training
from nachosv2.setup.utils_training import is_image_3D
from nachosv2.training.training_processing.partitions import generate_dict_folds_for_partitions
from nachosv2.training.training_processing.training_fold import TrainingFold
from nachosv2.modules.timer.precision_timer import PrecisionTimer
from nachosv2.modules.timer.write_timing_file import write_timing_file
from nachosv2.output_processing.memory_leak_check import initiate_memory_leak_check, end_memory_leak_check
from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.define_execution_device import define_execution_device
from nachosv2.setup.get_config import get_config
from nachosv2.checkpoint_processing.load_save_metadata_checkpoint import write_log
from nachosv2.training.hpo.hpo import get_hp_configuration

"""
Green: indications about where is the training
Cyan: verbose mode
Magenta: results
Yellow: warnings
Red: errors, fatal or not
"""


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
    print("Validation folds:", validation_fold)
            
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
            if config_dict['validation_fold_list'] is None:
                print(colored("For cross-tesing, validation_fold_list is not used"), "red")
            return [None]
        return normalize_to_list(fold_list) 
    else:
        if not fold_list:  # If fold_list is None or empty
            print(colored("Not fold provided for test fold. Using all folds in fold_list"), "red")            
            return config_dict['fold_list']
        
        return normalize_to_list(fold_list)


def get_gpu_index(num_gpus_per_device_to_use: int,
                  rank: int,
                  enable_dummy_node: bool):
    num_gpus_available = torch.cuda.device_count()
    
    print(colored(f'Rank {rank}', 'cyan'))
    print("Num GPUs Available: ", num_gpus_available)
    print("Num GPUs to use: ", num_gpus_per_device_to_use)

    index_gpu = -1

    if enable_dummy_node:
    
        # Assuming we discard one process
        # Assunming n_gpus 2
        # mod number is (# gpus +1)
        # Rank 0 Rank 1  Rank 2
        #        0       1  
        # Rank 3 Rank 4  Rank 5
        # 2      0       1
        # Rank 6 Rank 7  Rank 8 
        # 2      0       1      
        
        # The value 2 will do nothing 
        index_gpu = (rank-1) % (num_gpus_per_device_to_use+1)
        print("index_gpu =", index_gpu)
    else:
        index_gpu = (rank-1) % num_gpus_per_device_to_use
    
    return index_gpu


def fill_list_with_dummy_processes(n_proc: int,
                                   num_gpus_per_device_to_use: int):                               
    # r: rank
    # 2 GPUs
    #           | r0 r1 r2 | r3 r4 r5 | r6 r7 r8 | r9 r10 r11 |
    # index_gpu       0  1 |  2  0  1 |  2  0  1 |  2   0   1 |
    #                         ^          ^          ^           
    # ranks with index_gpu are not used
    # r3, r6, r9 
    
    #                   # processes 
    # n_unused_ranks = -------------- - 1
    #                   # gpus + 1
    # The number of unused ranks is calculated by the formula above
    # we first divided by the number of of gpus per device to use
    # plus one ( because of dummy node)
    # me made a substraction because process rank 0 is the 
    # manager process
        
    n_unused_ranks = int(n_proc/(num_gpus_per_device_to_use+1) - 1)
    
    return [(num_gpus_per_device_to_use+1)*i for i in range(1, n_unused_ranks+1)]


def determine_if_cv_loop(loop: str) -> bool:
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

    if len(execution_device_list) > 1:
        raise ValueError("For sequential training, only one device can be used.")

    # # Checks for memory leaks
    # memory_leak_check_enabled = False
    # # TODO: verify this memory check
    # if memory_leak_check_enabled:
    #     memory_snapshot = initiate_memory_leak_check()

    is_cv_loop = determine_if_cv_loop(loop)

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

    indices_loop_list = create_loop_indices(config_dict,
                                            is_cv_loop)

    n_combinations = len(indices_loop_list)

    # Starts a timer
    training_timer = PrecisionTimer()

    for index, indices_loop_dict in enumerate(indices_loop_list):
        perform_single_training(index=index,
                                n_combinations=n_combinations,
                                indices_loop_dict=indices_loop_dict,
                                is_cv_loop=is_cv_loop,
                                df_metadata=df_metadata,
                                execution_device=execution_device_list[0],
                                config_dict=config_dict,
                                is_verbose_on=is_verbose_on)
        
    # Stops the timer and prints the elapsed time
    elapsed_time_seconds = training_timer.get_elapsed_time()
    print(colored(f"\nElapsed time: {elapsed_time_seconds:.2f} seconds.", 'magenta'))

    # Creates a file and writes elapsed time in it
    # Make the next line fit in 80 characters
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
    
    num_gpus_per_device_to_use = len(execution_device_list)
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_proc = comm.Get_size()
    
    # Rank 0 initializes the program and runs the configuration loops
    if rank == 0:
        is_cv_loop = determine_if_cv_loop(loop)

        
        # Double-checks that the validation subjects are unique
        if is_cv_loop:  # Only if we are in the inner loop
            check_unique_subjects(config_dict["validation_fold_list"],
                                  "validation")

        if is_verbose_on:  # If the verbose mode is activated
            print(colored("Double-checks of test and validation uniqueness successfully done.", 'cyan'))

        indices_loop_list = create_loop_indices(config_dict,
                                                is_cv_loop)
        
        n_tasks = len(indices_loop_list)

        # No configs, no run
        if n_tasks == 0:
            # TODO: determine if sending False is necesssary
            for subrank in range(1, n_proc):
                comm.send(False, dest=subrank)
            raise ValueError(colored("No configurations given.", 'yellow'))

        path_csv_metadata = config_dict["path_metadata_csv"]
        df_metadata = read_metadata_csv(path_csv_metadata)
        
        # Listen for process messages while running
        process_finished_list = []
        # add manually when b_dummy is used 
        # to avoid problems later
        if enable_dummy_process:
            dummy_process_list = fill_list_with_dummy_processes(
                                    n_proc,
                                    num_gpus_per_device_to_use)
            process_finished_list.extend(dummy_process_list)
            print("process_finished_list =", process_finished_list)
        
        # reverse order to pop the last element
        indices_loop_list = indices_loop_list[::-1]
        
        # Starts a timer
        training_timer = PrecisionTimer()
        
        while True:
            # it received rank from other processes
            subrank = comm.recv(source=MPI.ANY_SOURCE)
            
            for index, indices_loop_dict in enumerate(indices_loop_list):
                dict_to_send = { "index": index,
                                 "n_combinations": n_tasks,
                                 "indices_loop_dict": indices_loop_dict,
                                 "is_cv_loop": is_cv_loop,
                                 "df_metadata": df_metadata,
                                 "config_dict": config_dict,
                                 "is_verbose_on": is_verbose_on}
                comm.send(dict_to_send, dest=subrank)

            # Send task if the process is ready
            # if indices_loop_list:
            #     print(colored(f"Rank 0 is sending rank {subrank} its task "
            #                 f"{len(indices_loop_list)}/{n_tasks}.", 'green'))
            #     comm.send(indices_loop_list.pop(), dest=subrank)
            #     next_task_index += 1

            print(colored(f"Rank 0 is terminating rank {subrank}, no tasks to give.", 'red'))
            comm.send(False, dest=subrank)
            process_finished_list += [subrank]
            
            # Stops the timer and prints the elapsed time
            elapsed_time_seconds = training_timer.get_elapsed_time()
            print(colored(f"\nElapsed time: {elapsed_time_seconds:.2f} seconds.", 'magenta'))

            # Creates a file and writes elapsed time in it
            # Make the next line fit in 80 characters
            loop_folder = "CT" if not is_cv_loop else "CV"
            timing_directory_path = Path(config_dict["output_path"]) / loop_folder /"training_timings" # The directory's path where to put the timing file
            timing_directory_path.mkdir(mode=0o775, parents=True, exist_ok=True)
            
            write_timing_file(training_timer,
                              timing_directory_path,
                              config_dict,
                              is_verbose_on)
            
    # rank > 0
    else: 
        
        index_gpu = get_gpu_index(num_gpus_per_device_to_use,
                                  rank,
                                  enable_dummy_process)
        
        # Listen for the first task
        print("num_gpus_per_device_to_use:", num_gpus_per_device_to_use)
        print(f"rank: {rank}, index_gpu: {index_gpu}")
        print("enable_dummy_process:", enable_dummy_process)
        
        if index_gpu != -1:

            comm.send(rank, dest=0)

            print(colored(f'Rank {rank} is listening for process 0.', 'cyan'))
            task = comm.recv(source=0)
            
            # While there are tasks to run, train
            while task:
                        
                # Training loop       
                perform_single_training(
                        index=task["index"],
                        n_combinations=task["n_combinations"],
                        indices_loop_dict=task["indices_loop_dict"],
                        is_cv_loop=task["is_cv_loop"],
                        df_metadata=task["df_metadata"],
                        execution_device=f"cuda:{index_gpu}",  # Use the GPU assigned to this rank
                        config_dict=task["config_dict"],
                        is_verbose_on=task["is_verbose_on"])
                
                comm.send(rank, dest=0)
                task = comm.recv(source=0)
                
            # Nothing more to run.
            print(colored(f'Rank {rank} terminated. All jobs finished for this process.', 'yellow'))


def train():

    args = parse_command_line_args()

    # Defines the arguments
    config_dict = get_config(args['file'])
    enable_parallelization = args['parallel']
    execution_device_list = args['devices']
    is_verbose_on = args['verbose']
    enable_dummy_process = args['enable_dummy_process']
    loop = args["loop"]

    # Sequential implementation
    if not enable_parallelization:
        train_sequential(config_dict,
                         execution_device_list,
                         is_verbose_on,
                         loop)
    # Parallel implementation
    else:
        pass
# Sequential Inner Loop
if __name__ == "__main__":
    train()
