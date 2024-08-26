import os
import random
from termcolor import colored


def get_image_path_list(data_input_directory_path, bool_shuffle_images, seed, is_verbose_on = False):
    """
    Gets all of the input paths of the image data.

    Args:
        data_input_directory_path (str): A path to some directory.
        bool_shuffle_images (bool): Whether to shuffle the image paths or not.
        seed (int): A random seed.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)

    Raises:
        Exception: If an input path cannot be reached.

    Returns:
        image_path_list (list of str): A list of image paths.
    """
    
    # Sees if it exists
    if not os.path.isdir(data_input_directory_path):
        raise Exception(colored(f"Error: '{data_input_directory_path}' is not a valid input path.", 'red'))
    
    
    # Creates the empty path list
    image_path_list = []
    
    # Fills the path list
    _recursively_get_all_images_path(data_input_directory_path, image_path_list)
    
    
    # Sorts the path list        
    image_path_list.sort()
    
    
    # Shuffles the path list if asked
    if bool_shuffle_images:
        _shuffle_image_path_list(seed, image_path_list)


    if is_verbose_on: # If the verbose mode is activated
        print(colored('Finished getting the image paths.', 'cyan'))
    
    
    return image_path_list


  
def _recursively_get_all_images_path(data_input_directory_path, image_path_list):
    """
    Recursively gets the paths of ALL images within a directory and its subdirectories.

    Args:
        path (str): A path to some directory.
        image_path_list (list of str): A list of paths to images.
    """
    
    # For each file or sub-directory in the data directory 
    for item in os.listdir(os.path.abspath(data_input_directory_path)):
        
        # Creates its full path
        full_path = os.path.join(data_input_directory_path, item)
        
        # If the item is a file
        if os.path.isfile(full_path):
            
            # If the file is an image file
            if full_path.endswith((".png", ".jpg", ".jpeg", ".tiff")):
                image_path_list.append(full_path) # Adds its path to the list
            
            # If the file is an non-image file
            else:
                print(colored(f"Warning: Non-image file detected: {full_path}", 'yellow'))
        
        # If the item is a folder     
        else:
            _recursively_get_all_images_path(full_path, image_path_list)



def _shuffle_image_path_list(seed, image_path_list):
    """
    Shuffles the list of image's paths
    
    Args:
        seed (int): Random seed.
        image_path_list (list of str): A list of paths to images.
    """
    
    # Sets the seed
    if seed:
        random.seed(seed)
    
    # Shuffles the images
    random.shuffle(image_path_list)
    