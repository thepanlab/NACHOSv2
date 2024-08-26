import numpy as np


def split_dataset(dataset, number_of_splits):
    """
    Split a dataset into a specified number of random subsets.
    
    Parameters:
    - dataset: The dataset to split.
    - number_of_splits: The number of splits to create.
    
    Returns:
    - A list of arrays, each containing indices for one split.
    """
    
    # Initialization
    splits = []
    
    
    # Gets the size of the dataset
    data_size = len(dataset)
    
    
    # Creates an array of indices and shuffles it
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    
    
    # Calculates the size of each split
    split_size = data_size // number_of_splits

    
    for i in range(number_of_splits):
        
        # Calculates the start index of the current split
        start_index = i * split_size
        
        # Calculates the end index of the current split
        end_index = (i + 1) * split_size
        
        # Slices the indices array to get the current split
        split_indices = indices[start_index:end_index]
        
        # Appends the current split to the splits list
        splits.append(split_indices)
    

    return splits
