from torchvision import transforms


class NormalizeTransform:
    def __init__(self, mean, std):
        """
        Creates the normalization.
        
        Args:
            mean (float): The mean of the dataset.
            std (float): The standard deviation of the dataset.
        """
        
        self.mean = mean
        self.std = std



    def __call__(self, tensor):
        """
        Normalizes the data.
        
        Args:
            tensor (PyTorch tensor): The image to normalize.
        """
        
        return transforms.functional.normalize(tensor, self.mean, self.std)
    