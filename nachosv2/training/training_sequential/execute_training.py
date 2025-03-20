from typing import List
import itertools
import random
from termcolor import colored
from mpi4py import MPI
import pandas as pd
import torch
from nachosv2.data_processing.check_unique_subjects import check_unique_subjects
from nachosv2.data_processing.read_metadata_csv import read_metadata_csv
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


def create_loop_indices(test_fold_list: List[str],
                        hpo_list: List[dict],
                        validation_fold_list: List[str]) -> List[dict]:
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
    list_loop_indices = []
    for t, h, v in itertools.product(test_fold_list,
                                     hpo_list,
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


    # Defines if the images are 2D or 3D
    is_3d = define_dimensions(config_dict)
    
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


def execute_training(execution_device_list: List[str],
                     config_dict: dict,
                     is_cv_loop: bool = True,
                     enable_parallelization: bool = False,
                     is_verbose_on: bool = False):
    """
    Runs the sequential training process for each configuration and test subject.

    Args:
        execution_device (str): The name of the device that will be use.
        config_dict (dict): configuration file.
        is_cv_loop (bool): If this is the cross-validation loop. (Optional)        
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
    
    Raises:
        Exception: When there are repeated test subjects or validation subjects
    """

              


    # Creates the path to the data CSV files
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
    
    validation_fold_list = config_dict.get('validation_fold_list', [None])
    
    if is_verbose_on:  # If the verbose mode is activated
        print(colored("Double-checks of test and validation uniqueness successfully done.", 'cyan'))
    
    
    # Creates test_subjects_list as a list, regardless of its initial form
    if not config_dict['test_fold_list']: # If the test_subjects list is empty, uses all subjects
        test_fold_list = config_dict['fold_list']
    elif isinstance(config_dict['test_fold_list'], str): # If the test_subjects list is a string, uses the corresponding list
        test_fold_list = [config_dict['test_fold_list']]
    elif isinstance(config_dict['test_fold_list'], list): # If the test_subjects list is a list, uses it
        test_fold_list = config_dict['test_fold_list']

    if config_dict["use_hpo"]:
        random.seed(config_dict['seed_hpo'])
    
    hpo_configurations = get_hpo_configuration(config_dict)    
    
    indices_loop_list = create_loop_indices(test_fold_list,
                                            hpo_configurations,
                                            validation_fold_list)
    
    n_combinations = len(indices_loop_list)
   
   # Sequential implementation
    if not enable_parallelization:
        for index, indices_loop_dict in enumerate(indices_loop_list):
            perform_single_training(index=index,
                                    n_combinations=n_combinations,
                                    indices_loop_dict=indices_loop_dict,
                                    is_cv_loop=is_cv_loop,
                                    df_metadata=df_metadata,
                                    execution_device=execution_device_list[0],
                                    config_dict=config_dict,
                                    is_verbose_on=is_verbose_on)
    # Parallel implementation
    else:
    # TODO
    # Parallelization shoudl be done here
       # Initialize MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        n_proc = comm.Get_size()
    
        # Initalize TF, set the visible GPU to rank%2 for rank > 0
        # tf_config = tf.compat.v1.ConfigProto()
        if rank != 0:
            num_gpus = torch.cuda.device_count()

            print(colored(f'Rank {rank}', 'cyan'))
            print("Num GPUs Available: ", num_gpus)
            print("GPUs Available: ", torch.cuda.get_device_name(rank))        
        
            index_gpu = -100
        
            if b_dummy:
            
                # Assuming we discard one process
                # Assunming # gpus 2
                # mod number is (# gpus +1)
                # Rank 0 Rank 1  Rank 2
                #        0       1  
                # Rank 3 Rank 4  Rank 5
                # 2      0       1
                # Rank 6 Rank 7  Rank 8 
                # 2      0       1      
                
                # The value 2 will do nothing 
                index_gpu = (rank-1) % (n_gpus+1)
                print("index_gpu =", index_gpu)
            else:
                index_gpu = (rank-1) % n_gpus
        
    # Rank 0 initializes the program and runs the configuration loops
    if rank == 0:  
        
        # Get start time
        start_time_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()
        start_perf = time.perf_counter()
        
        # Get the configurations
        configs = parse_training_configs(config_loc)
        n_configs = len(configs)
        next_task_index = 0
        
        # No configs, no run
        if n_configs == 0:
            print(colored("No configurations given.", 'yellow'))
            for subrank in range(1, n_proc):
                comm.send(False, dest=subrank)
            exit(-1)
        
        # Get the tasks for each process
        tasks = split_tasks(configs, is_outer)
        # tasks is a list of tuples
        # where tuple has 4 elements
        # 0: dictionary of configuration of hyperparameters
        # 1: number of epochs
        # 2: test_fold
        # 3: validation_fold
        print("len(tasks(top3)) = ", len(tasks[:3]))
        print("tasks(top3) = ", tasks[:3])

        n_tasks = len(tasks)
        
        # Listen for process messages while running
        exited = []
        # add manually when b_dummy is used 
        # to avoid problems later
        if b_dummy:
            
            # r: rank
            # 2 GPUs
            #       | r0 r1 r2 | r3 r4 r5 | r6 r7 r8 | r9 r10 r11 |
            # index_gpu   0  1 |  2  0  1 |  2  0  1 |  2   0   1 |
            #                     ^          ^          ^           
            # ranks with index_gpu are not used
            # r3, r6, r9 
            
            n_not_used_ranks = int(n_proc/(n_gpus+1) - 1)
            
            exited.extend([(n_gpus+1)*i for i in range(1, n_not_used_ranks+1)])
            
        print("exited =", exited)

        while True:
            # it received rank from other processes
            subrank = comm.recv(source=MPI.ANY_SOURCE)
            
            # Send task if the process is ready
            if tasks:
                print(colored(f"Rank 0 is sending rank {subrank} its task {len(tasks)}/{n_tasks}.", 'green'))
                comm.send(tasks.pop(), dest=subrank)
                next_task_index += 1
                    
            # If no task remains, terminate process
            else:
                print(colored(f"Rank 0 is terminating rank {subrank}, no tasks to give.", 'red'))
                comm.send(False, dest=subrank)
                exited += [subrank]
                
                # Check if any processes are left, end this process if so
                if all(subrank in exited for subrank in range(1, n_proc)):
                    
                    # Get end time and print
                    print(colored(f"Rank 0 is printing the processing time.", 'red'))
                    elapsed_time = time.perf_counter() - start_perf

                    if not os.path.exists("../results/training_timings"):
                        os.makedirs("../results/training_timings")
                    outfile = f'_TIME_MPI_OUTER_{start_time_name}.txt' if is_outer else f'_TIME_MPI_INNER_{start_time_name}.txt'
                    with open(os.path.join("../results/training_timings", outfile), 'w') as fp:
                        fp.write(f"{elapsed_time}")
                    print(colored(f'Rank {rank} terminated. All other processes are finished.', 'yellow'))
                    break
            
    # The other ranks will listen for rank 0's messages and run the training loop
    else: 
        # tf.config.run_functions_eagerly(True)
        
        # Listen for the first task
        print("n_gpus:", n_gpus)
        print(f"rank: {rank}, index_gpu: {index_gpu}")
        print("b_dummy:", b_dummy)
        
        if not b_dummy or ( b_dummy and index_gpu != n_gpus):

            print(f"physical_devices[{index_gpu}]=", physical_devices[index_gpu])
            tf.config.set_visible_devices(physical_devices[index_gpu], 'GPU')
            tf.config.experimental.set_memory_growth(physical_devices[index_gpu], True)

            comm.send(rank, dest=0)

            print(colored(f'Rank {rank} is listening for process 0.', 'cyan'))
            task = comm.recv(source=0)
            
            # While there are tasks to run, train
            while task:
                        
                # Training loop
                config, n_epochs, test_subject, validation_subject = task
                print(colored(f"rank {rank}: test {test_subject}, validation {validation_subject}", 'cyan'))         
                
                run_training(rank, config, n_epochs, test_subject, validation_subject, is_outer)
                comm.send(rank, dest=0)
                task = comm.recv(source=0)
                
            # Nothing more to run.
            print(colored(f'Rank {rank} terminated. All jobs finished for this process.', 'yellow'))
