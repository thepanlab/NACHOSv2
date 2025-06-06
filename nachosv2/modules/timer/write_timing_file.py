from pathlib import Path
from termcolor import colored
from nachosv2.modules.timer.precision_timer import PrecisionTimer


def write_timing_file(timer: PrecisionTimer,
                      directory_path: Path,
                      config_dict: dict,
                      is_verbose_on: bool = False):
    """
    Creates a timing file and writes in it the elapsed time.
    
    Args:
        timer (PrecisionTimer): The timer that times the time you want to write.
        directory_path (str): The path to the directory where you want the file to be created.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
    """
    

    # Initializations
    elapsed_time = timer.get_elapsed_time()
    timer_name = timer.get_timer_name()
    
    timing_file = f"{config_dict['job_name']}"+ \
                    f"_TIME_SEQ_INNER_{timer_name}.txt"
    timing_path = directory_path / timing_file

    with open(timing_path, 'w') as file_pointer:
        file_pointer.write(str(elapsed_time))

    if is_verbose_on: # If the verbose mode is activated
        print(colored("Timing file succesfully created.", "cyan"))
