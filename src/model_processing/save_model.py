import torch


def save_model(model, path):
    '''
    Saves the model into the path.
    
    Args:
        model (TrainingModel): The model to save.
        path (str): The path where to save the model.
    '''

    torch.save(model.state_dict(), path)
