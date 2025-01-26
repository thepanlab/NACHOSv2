import torch.nn as nn
import torch.nn.init as init

def initialize_model_weights(model):
    """
    Initializes the weights of the given model
    
    Args:
        model (TrainingModel): The untrained model.
    """
    if isinstance(model, nn.Conv2d):
        init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            init.constant_(model.bias, 0)
    elif isinstance(model, nn.Linear):
        init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm2d):
        init.constant_(model.weight, 1)
        init.constant_(model.bias, 0)

    # if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
    #     nn.init.kaiming_normal_(model.weight, mode = 'fan_out',
    #                             nonlinearity = 'relu')
    #     if model.bias is not None:
    #         nn.init.constant_(model.bias, 0)
    