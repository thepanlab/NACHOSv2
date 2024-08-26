from termcolor import colored

from torch.optim import lr_scheduler


def create_early_stopping(optimizer, patience, is_verbose_on = False):
    """
    Creates the early_stopping.
    
    Args:
        optimizer (optim.SGD): The optimizer.
        patience (int): The number of epochs to wait before triggering early stopping.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
    """
    
    early_stopping = lr_scheduler.ReduceLROnPlateau(
        optimizer,              # The optimizer
        mode = 'min',           # Monitor validation loss metric
        patience = patience,    # Number of epochs to wait before triggering early stopping
        factor = 0.1,           # Reduces learning rate by 10x if metric does not improve
        verbose = True,         # Displaies information when reducing learning rate
        threshold = 0.0001,     # Criterion to determine if improvement is sufficient to not trigger early stopping
        threshold_mode = 'rel', # Applies threshold criterion relative to baseline value of metric
        cooldown = 0,           # Number of epochs to wait after reducing learning rate before resuming monitoring
        min_lr = 0,             # Lower bound of learning rate
        eps = 1e-08             # Epsilon term to avoid division by zero
    )
    
    
    if is_verbose_on: # If the verbose mode is activated
        print(colored("Early stopping created.", 'cyan'))
        
        
    return early_stopping
