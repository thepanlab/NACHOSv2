import os
import sys
from termcolor import colored

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from scripts.training.training_sequential.sequential_processing import sequential_processing
from src.modules.timer.precision_timer import PrecisionTimer
from src.modules.timer.write_timing_file import write_timing_file
from src.output_processing.memory_leak_check import initiate_memory_leak_check, end_memory_leak_check
from src.setup.command_line_parser import command_line_parser
from src.setup.define_execution_device import define_execution_device
from src.setup.get_training_configs_list import get_training_configs_list


"""
Green: indications about where is the training
Cyan: verbose mode
Magenta: results
Yellow: warnings
Red: errors, fatal or not
"""


# Sequential Inner Loop
if __name__ == "__main__":
    """
    Called when this file is run.
    """   
    
    # Parses the command line arguments
    command_line_arguments = command_line_parser()
    
    # Defines the arguments
    list_of_configs, list_of_configs_paths = get_training_configs_list(
        command_line_arguments['file'],
        command_line_arguments['folder'])
    
    is_verbose_on = command_line_arguments['verbose']
    wanted_execution_device = command_line_arguments['device']
    
    # Defines the execution device
    execution_device = define_execution_device(wanted_execution_device)
    
    
    # Starts a timer
    all_programm_timer = PrecisionTimer()
    
    
    # Checks for memory leaks
    check_memory_leak = False
    
    if check_memory_leak:
        snapshot1 = initiate_memory_leak_check()
    
    
    # Runs the job
    is_outer_loop = False
    sequential_processing(
        execution_device,
        list_of_configs,
        list_of_configs_paths,
        is_outer_loop,
        is_verbose_on
        )
    
    
    # Checks for memory leaks
    if check_memory_leak:
        end_memory_leak_check(snapshot1)
    
    
    # Stops the timer and prints the elapsed time
    elapsed_time = all_programm_timer.get_elapsed_time()
    print(colored(f"\nElapsed time: {elapsed_time} seconds.", 'magenta'))

    # Creates a file and writes elapsed time in it
    timing_directory_path = "../results/training_timings" # The directory's path where to put the timing file
    write_timing_file(all_programm_timer, timing_directory_path, is_verbose_on)
