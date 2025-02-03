import os
import sys
from termcolor import colored

from nachosv2.training.training_sequential.sequential_processing import sequential_processing
from nachosv2.modules.timer.precision_timer import PrecisionTimer
from nachosv2.modules.timer.write_timing_file import write_timing_file
from nachosv2.output_processing.memory_leak_check import initiate_memory_leak_check, end_memory_leak_check
from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.define_execution_device import define_execution_device
from nachosv2.setup.get_config_list import get_config_list


"""
Green: indications about where is the training
Cyan: verbose mode
Magenta: results
Yellow: warnings
Red: errors, fatal or not
"""


def run_training():
    # Parses the command line arguments
    args = parse_command_line_args()
    
    # Defines the arguments
    list_dict_configs = get_config_list(args['file'], args['folder'])
    
    is_verbose_on = args['verbose']
    wanted_execution_device = args['device']
    
    # Defines the execution device
    execution_device = define_execution_device(wanted_execution_device)
    
    # Starts a timer
    training_timer = PrecisionTimer()
    
    # Checks for memory leaks
    memory_leak_check_enabled = False
    
    if memory_leak_check_enabled:
        memory_snapshot = initiate_memory_leak_check()
    
    # Runs the job
    is_outer_loop = False
    
    sequential_processing(
        execution_device,
        list_dict_configs,
        is_outer_loop,
        is_verbose_on
        )

    # Checks for memory leaks
    if memory_leak_check_enabled:
        end_memory_leak_check(memory_snapshot)   
    
    # Stops the timer and prints the elapsed time
    elapsed_time_seconds = training_timer.get_elapsed_time()
    print(colored(f"\nElapsed time: {elapsed_time_seconds} seconds.", 'magenta'))

    # Creates a file and writes elapsed time in it
    # Make the next line fit in 80 characters
    timing_directory_path = "../results/training_timings" # The directory's path where to put the timing file
    write_timing_file(training_timer,
                      timing_directory_path,
                      is_verbose_on)


# Sequential Inner Loop
if __name__ == "__main__":
    run_training()
