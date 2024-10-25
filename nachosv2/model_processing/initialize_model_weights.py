import torch.nn as nn

def initialize_model_weights(model):
    """
    Initializes the weights of the given model
    
    Args:
        model (TrainingModel): The untrained model.
    """
    
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
        
        nn.init.kaiming_normal_(model.weight, mode = 'fan_out', nonlinearity = 'relu')
        
        if model.bias is not None:
            
            nn.init.constant_(model.bias, 0)
    