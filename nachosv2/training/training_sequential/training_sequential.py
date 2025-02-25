import os
import sys
from pathlib import Path
from termcolor import colored

from nachosv2.training.training_sequential.execute_training import execute_training
from nachosv2.modules.timer.precision_timer import PrecisionTimer
from nachosv2.modules.timer.write_timing_file import write_timing_file
from nachosv2.output_processing.memory_leak_check import initiate_memory_leak_check, end_memory_leak_check
from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.define_execution_device import define_execution_device
from nachosv2.setup.get_config import get_config


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
    config_dict = get_config(args['file'])
    
    is_verbose_on = args['verbose']
    
    # default "cuda:1" if not specified
    wanted_execution_device = args['device']
    
    # Defines the execution device
    execution_device = define_execution_device(wanted_execution_device)
    
    # Starts a timer
    training_timer = PrecisionTimer()
    
    # Checks for memory leaks
    memory_leak_check_enabled = False
    
    if memory_leak_check_enabled:
        memory_snapshot = initiate_memory_leak_check()
    
    # Default to is_cross_testing = False, i.e. cross-validation loop
    # is enabled
    loop = args["loop"]
    
    if loop == 'cross-testing':
        is_cv_loop = False
    else:
        is_cv_loop = True
        
    execute_training(
        execution_device,
        config_dict,
        is_cv_loop,
        is_verbose_on
        )

    # Checks for memory leaks
    if memory_leak_check_enabled:
        end_memory_leak_check(memory_snapshot)
    
    # Stops the timer and prints the elapsed time
    elapsed_time_seconds = training_timer.get_elapsed_time()
    print(colored(f"\nElapsed time: {elapsed_time_seconds:.2f} seconds.", 'magenta'))

    # Creates a file and writes elapsed time in it
    # Make the next line fit in 80 characters
    timing_directory_path = Path(config_dict["output_path"]) / "training_timings" # The directory's path where to put the timing file
    timing_directory_path.mkdir(mode=0o777, parents=True, exist_ok=True)
    
    write_timing_file(training_timer,
                      timing_directory_path,
                      config_dict,
                      is_verbose_on)

# Sequential Inner Loop
if __name__ == "__main__":
    run_training()
