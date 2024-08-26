from torch.utils.data import DataLoader


def calculate_mean_std(dataset):
    """
    Calculates the mean and the standard deviation.
    
    Args:
        dataset (Dataset): The dataset you want to calculate the mean and the standard deviation.
    
    Returns:
        mean (float): The mean of the dataset.
        std (float): The standard deviation of the dataset.
    """
    
    # Initializations
    mean = 0.
    std = 0.
    
    
    # Creates the data loader
    dataloader = DataLoader(
        dataset, batch_size = 100, num_workers = 4, shuffle = False
    )
    
    
    for images in dataloader:
        batch_samples = images.size(0)
        
        images = images.view(batch_samples, images.size(1), -1)
        
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    
    
    # Final calculus of the mean and the standard deviation
    mean /= len(dataset)
    std /= len(dataset)
    
    
    return mean, std
