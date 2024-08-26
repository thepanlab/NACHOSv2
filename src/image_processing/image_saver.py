import numpy as np
import skimage.io


def save_image(image, filename):
    """
    Saves a tensor image to a file.

    Args:
        image (Tensor): The image tensor to save.
        filename (str): The filename where the image will be saved.
    """
    
    # Converts the tensor into a NumPy array and reorganises the dimensions
    img_array = image.permute(1, 2, 0).numpy()
    
    
    # Ensures that the values are in the range [0, 1]
    img_array = np.clip(img_array, 0, 1)
    
    
    # Converts to 8-bit integers (0-255)
    img_array = (img_array * 255).astype(np.uint8)
    
    
    # Saves the image
    skimage.io.imsave(filename, img_array)
    