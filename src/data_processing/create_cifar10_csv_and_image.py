import os
import pandas as pd
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from src.image_processing.image_saver import save_image


def create_cifar10_csv_and_image(dataset, indices, filename, image_directory, create_images = True):
    """
    Creates a CSV file containing filenames and labels of images in a dataset split.

    Args:
        dataset (Dataset): The dataset from which to extract images.
        indices (list): The indices of the images to include in the CSV.
        filename (str): The filename for the CSV file.
        image_directory (str): The directory where the images will be saved.
        create_images (bool): Whether to save the images to the disk. Default is True.
    """
    
    # Initializes a list to hold the data rows
    data = []
    
    # Defines the columns for the CSV file
    columns = ['files', 'labels']
    

    for index in indices:
        
        # Extracts image and label for the current index
        image, label = dataset[index]
        
        # Generates a filename for the image
        image_filename = f"cifar_{index}.png"
        
        
        # Saves the image to the specified directory
        if create_images:
            save_image(image, os.path.join(image_directory, image_filename))
        
        
        # Creates a row with the image filename and label
        row = [image_filename, label]
        data.append(row)
    
    
    # Creates a DataFrame and saves it as a CSV file
    df = pd.DataFrame(data, columns = columns)
    
    df.to_csv(os.path.join(f'{PROJECT_ROOT}/data', filename), index = False)
    