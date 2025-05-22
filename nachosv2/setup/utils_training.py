from typing import Union, List, Tuple
import torch


FloatOrListFloats = Union[float, List[float]]
def get_mean_stddev(
    number_channels: int,
    dataloader: "torch.utils.data.DataLoader",
    device: str,       
    )->Tuple[FloatOrListFloats, FloatOrListFloats]:
    """
    Calculate the mean and standard deviation for grayscale (1 channel) or RGB (3 channels) datasets.

    Args:
        dataloader (DataLoader): DataLoader providing batches of images.
        
    Returns:
        Tuple[Union[float, List[float]], Union[float, List[float]]]: 
        (mean, std), where mean and std are scalars (float) for grayscale 
        or lists of floats for RGB.
    """

    total_sum = torch.zeros(number_channels)
    total_sum_squared = torch.zeros(number_channels)
    num_pixels = 0
    
    # Iterate through the dataset
    for images, _, _ in dataloader:
        images = images.to(device)
        batch_size, channels, height, width = images.shape

        images = images.view(batch_size, channels, -1)
        
        # Accumulate sum and squared sum
        total_sum += images.sum(dim=(0, 2))  # Sum over batch and spatial dimensions
        total_sum_squared += (images ** 2).sum(dim=(0, 2))  # Sum of squares
        num_pixels += batch_size * height * width  # Total number of pixels per channel

    # Calculate mean and standard deviation
    mean = total_sum / num_pixels
    stddev = (total_sum_squared / num_pixels - mean ** 2).sqrt()
        
    # Convert to scalars for grayscale or lists for RGB
    if number_channels == 1:
        return mean.item(), stddev.item()
    elif number_channels == 3:
        return mean.tolist(), stddev.tolist()
    else:
        raise ValueError("Number of channels must be 1 or 3.")


def get_files_labels(partition: str,
                     df_metadata: 'pandas.DataFrame',
                     test_fold: str,
                     validation_fold: str,
                     training_folds_list: list) -> tuple:   
    # Dictionary mapping partitions to their respective fold names or conditions
    partition_queries = {
        'test': f"fold_name == @test_fold",
        'validation': f"fold_name == @validation_fold",
        'training': f"fold_name in @training_folds_list"
    }

    # Validate the partition argument
    if partition not in partition_queries:
        raise ValueError(f"Invalid partition specified: {partition}."
                            " Must be 'test', 'validation', or 'training'.")

    # Query the DataFrame once per function call
    partition_query = partition_queries[partition]
    filtered_data = df_metadata.query(partition_query)

    # Extract files and labels lists
    files = filtered_data['absolute_filepath'].tolist()
    labels = filtered_data['label'].tolist()

    return files, labels


def create_empty_learning_rate_freq_step_history():
    """
    TODO
    """
    
    # Creates the base history
    lr_history = {'epoch': [], 'step': [],
                  'learning_rate': []}
    
    return lr_history


def create_empty_history(is_cv_loop: bool,
                         metrics_dictionary: dict):
    """
    TODO
    """
    
    # Creates the base history
    if is_cv_loop:
        history = {'training_loss': [], 'validation_loss': [],
                   'training_accuracy': [], 'validation_accuracy': [],
                   'learning_rate': []}
    else:
        history = {'training_loss': [], 'training_accuracy': [],
                   'learning_rate': []}
    
    # Adds the wanted metrics to the history dictionary
    # add_lists_to_history(history, metrics_dictionary)

    return history


def is_image_3D(configuration: dict) -> bool:
    """
    Defines if the images are 2D or 3D.
    
    Args:
        configuration (dict): The JSON configuration file.
        
    Return:
        is_3d (bool): The boolean that say of its 2D or 3D. False for 2D, True for 3D.
    """
    
    try:      
        # Counts the number of dimensions
        number_of_dimensions = len(configuration["image_size"])

        # The number of dimensions should be 2 or 3
        if number_of_dimensions not in [2,3]:
            raise ValueError("The number of image dimensions must be either 2 or 3.")

        # Defines if it's 2D or 3D
        if number_of_dimensions == 2:
            is_3d = False        
        else:
            is_3d = True

        return is_3d
    
    except ValueError as e:
        print(colored(e, 'red'))
        sys.exit()
