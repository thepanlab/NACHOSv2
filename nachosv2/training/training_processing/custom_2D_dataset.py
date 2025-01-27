from pathlib import Path
from typing import Optional, Callable
import skimage.io
import skimage.transform
from skimage.util import img_as_float32
from skimage.color import rgb2gray
import numpy as np

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms

from nachosv2.image_processing.image_crop import crop_image
# from nachosv2.image_processing.image_transformations import image_transformation_2D


def ToTensor(image, label, number_channels):
    """Convert ndarrays in sample to Tensors.
            class based from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
            Dataloader can automatically transform the data to tensor
            but better to make sure that the data is in the right format in the first place
    """
    
    if image.ndim == 2 and number_channels == 1:
        image = image[np.newaxis, :, :]
    elif image.ndim == 3 and number_channels == 3:
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
    
    return torch.from_numpy(image), torch.tensor(label, dtype=torch.int8)

class Dataset2D(Dataset):
    def __init__(self,
                 dictionary_partition: dict,
                 number_channels: int,
                 do_cropping: bool = False,
                 crop_box: tuple = None,
                 transform: Optional[Callable] = None):
        """
        Initializes a custom dataset object.
        
        Args:
            prefix (str): The file path prefix = the path to the directory where the images are.
            data_dictionary (dict of list): The data dictionary.
            
            channels (int): The number of channels in an image.
            
            do_cropping (bool): Whether to crop the image.
            crop_box (tuple of int): The dimensions after cropping.
            
            normalizer (transforms.Compose): The image normalizer.
        """
        
        self.dictionary_partition = dictionary_partition
        self.number_channels = number_channels
        self.do_cropping = do_cropping
        self.crop_box = crop_box
        self.transform = transform


    def __len__(self):
        """
        Returns the dataset len.
        """
        
        return len(self.dictionary_partition['files'])


    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset given an index.
        
        Args:
            index (int): The index of a sample in the dataset.
        
        Returns:
            image_tensor (PyTorch tensor): The sample from the dataset.
            label_index (int): The label of the sample.
            image_index (int): The index of the sample in all the dataset.
            file_path (str): The file path.
        """
        
        # Opens the image
        filepath = self.dictionary_partition['files'][index]
        image = skimage.io.imread(filepath)
        label = self.dictionary_partition['labels'][index]
        
        image = img_as_float32(image)
        # Transformes the image
        # conditional that image differ from number of dimensions
        if self.number_channels == 1 and image.ndim == 3:
            image = rgb2gray(image)
        
        # If asked, crops the image
        if self.do_cropping:
            image = crop_image(image, self.crop_box)

        # Converts the image into a tensor
        image = transforms.ToTensor()(image)

        if self.transform:
            # Error here
            image = self.transform(image)

        # Dataset loader will automatically convert label to torch.LongTensor
        # i.e. sign integer 64-bit integer
        
        return image, label, filepath


class Custom2DDataset(Dataset):
    def __init__(self, prefix, data_dictionary, channels, do_cropping, crop_box, normalizer):
        """
        Initializes a custom dataset object.
        
        Args:
            prefix (str): The file path prefix = the path to the directory where the images are.
            data_dictionary (dict of list): The data dictionary.
            
            channels (int): The number of channels in an image.
            
            do_cropping (bool): Whether to crop the image.
            crop_box (tuple of int): The dimensions after cropping.
            
            normalizer (transforms.Compose): The image normalizer.
        """
        
        self.prefix = prefix
        self.data_dictionary = data_dictionary
        self.file_list = data_dictionary['files']
        
        self.channels = channels
        
        self.do_cropping = do_cropping
        self.crop_box = crop_box
        
        self.normalizer = normalizer


    def __len__(self):
        """
        Returns the dataset len.
        """
        
        return len(self.file_list)



    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset given an index.
        
        Args:
            index (int): The index of a sample in the dataset.
        
        Returns:
            image_tensor (PyTorch tensor): The sample from the dataset.
            label_index (int): The label of the sample.
            image_index (int): The index of the sample in all the dataset.
            file_path (str): The file path.
        """
        
        # Gets the image path
        image_path = f"{self.prefix}/{self.file_list[index]}"
        
        # Opens the image
        image = skimage.io.imread(image_path)
        
        # Transformes the image
        image = transform_to_one_channel(image)        
        # image = image_transformation_2D(image)        
        
        # If asked, crops the image
        if self.do_cropping:
            image = crop_image(image, self.crop_box)

        # Converts the image into a tensor
        image_tensor = self.normalizer(image)

        # Gets the label
        label_index = self.data_dictionary['labels'][index]
        
        # Gets the index
        image_index = self.data_dictionary['indexes'][index]
        
        # Gets the file path
        file_path = self.file_list[index]
        
        return image_tensor, label_index, image_index, file_path
    