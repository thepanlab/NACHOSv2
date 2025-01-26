from termcolor import colored


def crop_image(image, crop_box, is_verbose_on = False):
    """
    Crops an image in the shape of the crop box given.
    
    Args:
        image (numpy.ndarray): The image to crop.
        crop_box (list of str): The box that contains the dimensions.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
        
    Returns:
        cropped_image (numpy.ndarray): The image cropped.
    """
    
    # Extracts the coordinates from crop box
    x1, y1, x2, y2 = crop_box
    
    
    # Crops the image
    cropped_image = image[y1:y2, x1:x2]
    
    
    if is_verbose_on: # If the verbose mode is activated
        print(colored("Crop done.", 'cyan'))
    
    
    return cropped_image



def create_crop_box(offset_height, offset_width, target_height, target_width, is_verbose_on = False):
    """
    Creates a box that contains crop dimensions.
    
    Args:
        offset_height (int): The height of the offset.
        offset_width (int): The width of the offset.
        target_height (int): The height wanted.
        target_width (int): The width wanted.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
    
    Returns:
        crop_box (tuple of int): The dimensions after cropping.
    """
    
    # Defines the dimensions with better names for understanding purpose
    upper = offset_height
    left = offset_width
    lower = upper + target_height
    right = left + target_width
    
    # Creates the crop box
    crop_box = (left, upper, right, lower)
    
    
    if is_verbose_on: # If the verbose mode is activated
        print(colored("Crop box created", 'cyan'))
    
    return crop_box
