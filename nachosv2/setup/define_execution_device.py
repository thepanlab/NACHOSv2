from termcolor import colored

import torch


def define_execution_device(execution_device_name):
    '''
    Defines the device where the program will be running.
    
    Args:
        execution_device_name (str): The name of the execution device wanted
        
    Returns:
        execution_device (str): The name of the execution device that will be use
    '''
    
    printing_color = 'green' # The default color for printing. Used if everything is good
    
    
    # If CUDA is asked
    if execution_device_name == "cuda:0" or "cuda:1":
        
        # Checks if CUDA is available
        if torch.cuda.is_available():
            
            if execution_device_name == "cuda:0": # If GPU0 is asked
                torch.cuda.set_device(0) # Sets the execution device 
                
            elif execution_device_name == "cuda:1": # If GPU1 is asked
                torch.cuda.set_device(1) # Sets the execution device
            
            execution_device = execution_device_name # Sets the execution device's name
            
            
        else: # If CUDA is not available
            print(colored("Non-fatal error: CUDA non available"), 'red')
            execution_device = "cpu" # Sets the execution device's name
            printing_color = 'red' # Change the prinintg color
    

    # Prints which execution device will be used
    print(colored(f"The model will be running on {execution_device}", printing_color))
    
    return execution_device
