from termcolor import colored
import torch


def define_execution_device(device_name):
    '''
    Defines the device where the program will be running.
    
    Args:
        device_name (str): The name of the execution device wanted
        
    Returns:
        execution_device (str): The name of the execution device that will be use
    '''
    
    print_color = 'green' # The default color for printing. Used if everything is good
    
    list_cuda_devices_allowed = ["cuda:0", "cuda:1"]
    # If CUDA is asked
    # Check if a CUDA device is requested and available
    if device_name in list_cuda_devices_allowed and torch.cuda.is_available():
        torch.cuda.device(device_name)  # Set the specified CUDA device
        execution_device = device_name  # Set execution device to requested CUDA device
    else:
        print(colored("Non-fatal error: CUDA not available, switching to CPU.", 'red'))
        execution_device = "cpu"  # Fallback to CPU
        print_color = 'red'
    

    # Prints which execution device will be used
    print(colored(f"The model will be running on {execution_device}", print_color))
    
    return execution_device
