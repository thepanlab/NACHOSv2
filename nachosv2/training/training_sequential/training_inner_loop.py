import os
import sys
from termcolor import colored

from nachosv2.training.training_sequential.sequential_processing import sequential_processing
from nachosv2.modules.timer.precision_timer import PrecisionTimer
from nachosv2.modules.timer.write_timing_file import write_timing_file
from nachosv2.output_processing.memory_leak_check import initiate_memory_leak_check, end_memory_leak_check
from nachosv2.setup.command_line_parser import command_line_parser
from nachosv2.setup.define_execution_device import define_execution_device
from nachosv2.setup.get_training_configs_list import get_training_configs_list

"""
Green: indications about where is the training
Cyan: verbose mode
Magenta: results
Yellow: warnings
Red: errors, fatal or not
"""


def run_training():
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


# Sequential Inner Loop
if __name__ == "__main__":
    run_training()