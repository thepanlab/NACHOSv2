import torch


def custom_softmax(x):
    """
    Applies a custom softmax function on the data outputed by the model.
    
    Arg:
        x (PyTorch tensor): The PyTorch tensor containing the model's output data.
    
    Returns:
        softmax_x (PyTorch tensor): The PyTorch tensor containing the normalized and softmaxed model's output data.
    """
    
    # Normalizes
    normalized_x = _normalize(x)
    
    # Applies the stable softmax function
    softmax_x = _stable_softmax(normalized_x)
    
    
    return softmax_x



def _normalize(x):
    """
    Normalizes the model's output data.
    
    Arg:
        x (PyTorch tensor): The PyTorch tensor containing the model's output data.
    
    Returns:
        normalized_x (PyTorch tensor): The PyTorch tensor containing the normalized model's output data.
    """
    
    # Normalizes
    normalized_x = (x - torch.mean(x)) / torch.std(x)
    
    
    return normalized_x
    
    

def _stable_softmax(x):
    """
    Applies the stable softmax function on the model's output data.
    
    Arg:
        x (PyTorch tensor): The PyTorch tensor containing the model's output data.
    
    Returns:
        softmax_x (PyTorch tensor): The PyTorch tensor containing the softmaxed model's output data.
    """
    
    # Creates the softmax numerator
    z = x - x.max(dim = 1, keepdim = True)[0]
    numerator = torch.exp(z)
    
    # Creates the softmax denominator
    denominator = torch.sum(numerator, dim = 1, keepdim = True)
    
    # Applies the softmax
    softmax_x = numerator / denominator
    
    
    return softmax_x
