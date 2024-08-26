import os
from termcolor import colored

import torch


class Checkpointer():
    def __init__(self, n_epochs, k_epochs, file_name, mpi_rank, save_path = "./"):
        super().__init__()
        """
        This will create a custom checkpoint for every k epochs.
        
        Args:
            n_epochs (int): The total expected epochs.
            k_epochs (int): The interval of epochs at which to save.
            file_name (str): The model metadata. Contains the job name, config name, and subjects.
            
            mpi_rank (int): An optional value of some MPI rank. Default is none. (Optional)
            save_path (str): Where the checkpoint is saved to. Defaults to "./". (Optional)
        """ 
        
        self.n_epochs = n_epochs
        self.k_epochs = k_epochs
        self.file_name = file_name
        self.mpi_rank = mpi_rank
        self.save_path = save_path
        
        self.prev_save = None
    
    
        
    def on_epoch_end(self, epoch, model):
        """
        After each epoch, this function is called. Checkpoints are saved as <NAME>_<EPOCH>.h5
        
        Args:
            epochs (int): The current epoch.
            model (torch.nn.Module): The model to load the state_dict into.
        """ 
        
        # If the epoch is within k steps, saves it
        if epoch % self.k_epochs == 0 or (epoch+1) == self.n_epochs:
            
            # If the directory does not exist, creates it
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
    
            # Saves the checkpoint to file: Name_Current-epoch.pth
            new_save_path = os.path.join(self.save_path, f"{self.file_name}_{epoch + 1}.pth")
            torch.save(model.state_dict(), new_save_path)
            
            print(colored(f"Saved a checkpoint for epoch {epoch + 1}/{self.n_epochs}.", 'cyan'))
            
            # Keeps only the previous checkpoint for the most recent training fold, to save memory.
            self.clear_prev_save(new_save_path)



    def clear_prev_save(self, new_save_path = None):
        """
        Clears the most recent saved checkpoint in this training job to save space.
        
        Args:
            new_save_path (str): The current, new checkpoint location. (Optional)
        """ 

        # If a previous save exists, deletes it
        if self.prev_save and os.path.exists(self.prev_save):
            os.remove(self.prev_save)
        
        if new_save_path:
            self.prev_save = new_save_path
            
        else:
            self.prev_save = None



def get_most_recent_checkpoint(save_path, file_prefix, model, get_epoch = True):
    """
    Get the most recent checkpoint of a job, being the one with the highest epoch value.
    
    Args:
        save_path (str): Where the checkpoints are saved to.
        file_prefix (str): The job name in the file: <NAME>_<EPOCH>.h5
        model (torch.nn.Module): The model to load the state_dict into.
        get_epoch (bool): Whether to return the epoch count from the file.
    
    Returns:
        model (torch.nn.Module): The model with loaded state_dict.
        max_epoch (int): The epoch number if get_epoch is True. (Optional)
    """ 
    
    # Gets what checkpoints are in the given save path.
    checkpoints = [os.path.join(save_path, file) for file in os.listdir(save_path) if file.startswith(file_prefix)]
    
    if len(checkpoints) == 0:
        return None
    
    elif len(checkpoints) > 1:
        print(colored("Warning: Multiple checkpoints exist with the same file name prefix. Using the one with the greatest epoch...", 'yellow'))
    
    # Reads the epochs from each checkpoint and choose the maximum one.
    checkpoint_epochs = [int(os.path.splitext(os.path.basename(chkpt))[0].split('_')[-1]) for chkpt in checkpoints]
    max_epoch = max(checkpoint_epochs)
    model = load_checkpoint(checkpoints[checkpoint_epochs.index(max_epoch)], model, False)
    
    if not get_epoch:
        max_epoch = None
    
    return model, max_epoch



def load_checkpoint(path, model, get_epoch = True):
    """
    Loads a model from a PyTorch checkpoint.

    Args:
        path (str): The path to the checkpoint.
        model (torch.nn.Module): The model to load the state_dict into.
        get_epoch (bool): Whether to return the epoch number from the checkpoint name. Default is True. (Optional)

    Returns:
        model (torch.nn.Module): The model with loaded state_dict.
        epoch (int): The epoch number if get_epoch is True. (Optional)

    Raises:
        Exception: When no checkpoint exists.
    """
    
    # Gets the model from some path.
    if not os.path.exists(path):
        raise Exception(colored(f"Error: No checkpoint was found at '{path}'", 'red'))
    
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint)
    
    if get_epoch:
        epoch = path.split('/')[-1].split('.')[0].split('_')[-1] -1
    
    else:
        epoch = 0
    
    return model, epoch
