from termcolor import colored

from nachosv2.image_processing.image_utils import *
from nachosv2.file_processing.file_name_utils import get_file_name, remove_file_extension
from nachosv2.image_processing.image_crop import crop_image


def parse_image(filename, class_names, channels, do_cropping, offset_height, offset_width, target_height, target_width, label_position=None, use_labels=True): 
    """ Parses an image from some given filename and various parameters.
        
    Args:
        filename (Tensor str): A tensor of some file name.
        class_names (list of str): A list of label class names.
        
        channels (int): Channels in which to decode image. 
        do_cropping (bool): Whether to crop the image.
        offset_height (int): Image height offset.
        
        offset_width (int): Image width offset.
        target_height (int): Image height target.
        target_width (int): Image width target.
        
        label_position (int): Optional. The position of the label in the image name. Must provide if using labels. Default is None.
        use_labels (bool): Optional. Whether to consider/output the true image label. Default is True.
        
    Returns:
        (Tensor image): An image.
        (Tensor str): The true label of the image.
    """

    # Transforms the image
    image_path = get_file_name(filename)
    path_substring = remove_file_extension(image_path)
    image = read_image(filename)
    image = decode_image(image, channels = channels)


    if do_cropping:
        try:
            image = crop_image(image, offset_height, offset_width, target_height, target_width)
            
        except:
            raise ValueError(colored('Cropping bounds are invalid. Please review target size and cropping position values.', 'red'))
    
    
    # Finds the label if needed
    if use_labels:
        if not label_position or label_position < 0:
            raise ValueError(colored("Error: A label position is needed to parse the image class.", 'red'))
        
        path_label = extract_label_from_path(path_substring, "_", label_position)

        label_bool = get_label_from_class_names(path_label, class_names)
        
    
    return image, label_bool
