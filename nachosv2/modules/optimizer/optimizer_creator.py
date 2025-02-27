from termcolor import colored

import torch.optim as optim


def create_optimizer(model,
                     hyperparameters: dict,
                     is_verbose_on = False):
    '''
    Creates and returns a custom optimizer
    
    Args:
        model (TraininModel): The model.
        hyperparameters (dict): The dictionary of hyperparameters.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
        
    Returns:
        optimizer (torch.optim.SGD): The custom optimizer.
    '''
    
    
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparameters['learning_rate'], 
        momentum=hyperparameters.get('momentum',0), 
        nesterov=hyperparameters['enable_nesterov']
    )
    
    
    if is_verbose_on: # If the verbose mode is activated
        print(colored("Custom optimizer created.", 'cyan'))
    
    
    return optimizer
