from PIL import Image

import torch
import torchvision.transforms as transforms


def read_image(image_path):
    """
    Reads an image.
    
    Args:
        image_path (str): The path to the image to read.
    """
    
    return Image.open(image_path)



def decode_image(image, channels = 3):
    """
    Decodes an image.
    
    Args:
        image_path (str): The path to the image to read.
        channels (int): Teh number of channel. Default is 3. (Optional)
    """
    
    # Converts the image depending of the number of channels.
    if channels == 3: # 3 channels = RGB
        image = image.convert('RGB')
        
    elif channels == 1: # 1 channel = grey levels
        image = image.convert('L')
    
    
    # Converts the image to a tensor
    image = transforms.ToTensor()(image)
    
    return image



def extract_label_from_path(path_substring, delimiter = "_", label_position = 0): # Temp

    parts = path_substring.split(delimiter)
    if label_position < len(parts):
        return parts[label_position]
    else:
        raise IndexError(f"label_position {label_position} is out of range for the split parts.")



def get_label_from_class_names(path_label, class_names): # TODO change
    # Créer un tenseur booléen pour les noms de classes
    label_bool = torch.tensor([path_label == class_name for class_name in class_names], dtype=torch.float32)
    
    return torch.argmax(label_bool).item()