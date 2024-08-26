import skimage.io
import skimage.transform
from termcolor import colored

import torch


def image_transformation_2D(image, verbose = False):
    """
    Applies transformations to a 2D image.

    This includes converting grayscale images to RGB, resizing the image, and converting it to a PyTorch tensor.

    Args:
        image (numpy.ndarray): The input 2D image.
        verbose (bool): True to activate verbose mode. (Optional)

    Returns:
        image (torch.Tensor): The transformed image as a PyTorch tensor.
    """
    
    # If grayscale, converts to RGB
    if image.ndim == 2:
        image = skimage.color.gray2rgb(image)

    
    # Resizes the image
    image = skimage.transform.resize(image, (299, 299), anti_aliasing = True)
    
    
    # Converts to PyTorch tensor and changes dimension order
    image = torch.from_numpy(image).float().permute(2, 0, 1)
    
    
    if verbose:
        print(colored('Image transformed.', 'cyan'))
    
    
    return image



def image_transformation_3D(image, verbose = False):
    """
    Applies transformations to a 3D image.

    This includes converting grayscale images to RGB and normalizing the image.

    Args:
        image (numpy.ndarray): The input 3D image with shape (Channels, Depth, Height, Width).
        verbose (bool): True to activate verbose mode. (Optional)

    Returns:
        image (torch.Tensor): The transformed image as a PyTorch tensor.
    """
    
    """
    # If grayscale
    if image.shape[0] == 1:
        # Converts to RGB
        image = skimage.color.gray2rgb(image[0])
        # Changes the order of dimensions to (Channels, Depth, Height, Width)
        image = image.transpose((3, 0, 1, 2))
    
    else:
        # Ensures the order of dimensions is (Channels, Depth, Height, Width)
        image = image.transpose((0, 1, 2, 3))
    """
    
    # Converts to PyTorch tensor and changes dimension order
    image = torch.from_numpy(image).float()
    
    
    # Normalizes to [-1, 1]
    image = (image - 0.5) / 0.5
    
    
    if verbose:
        print(colored('Image transformed.', 'cyan'))
    
    
    return image