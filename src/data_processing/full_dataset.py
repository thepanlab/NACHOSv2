import os
import pandas as pd
import skimage.io
import skimage.transform

import torch
from torchvision import transforms


class FullDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, directory_path, csv_files, is_3d = False):
        """
        Initializes a custom dataset object.
        
        Args:
            prefix (str): The file path prefix = the path to the directory where the images are.
            directory_path (str): The path to the CSV directory.
            csv_files (list of str): The list of paths to the images.
        """

        self.prefix = prefix
        self.data = pd.DataFrame()
        self.is_3d = is_3d
        
        
        # Initializes the data
        for csv_file in csv_files:
            dataframe = pd.read_csv(os.path.join(directory_path, csv_file))
            self.data = pd.concat([self.data, dataframe], ignore_index = True)
        
        
        # Initializes the transform
        self.transform = transforms.Compose([
            transforms.Resize((229, 229)),  
            transforms.ToTensor()
        ])



    def __len__(self):
        """
        Returns the dataset len.
        """
        
        return len(self.data)



    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset given an index.
        
        Args:
            index (int): The index of a sample in the dataset.
        
        Returns:
            transformed_image (PyTorch tensor): The sample from the dataset.
        """
        
        # Initialization
        image_path = self.data.iloc[index]['files']

        # Makes sure the path is an absolut path
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.prefix, image_path)
        
        
        # Opens the image
        image = skimage.io.imread(image_path)
        
        # If grayscale, converts to RGB
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)

        
        # Transforms the image
        if self.is_3d:
            # Traitement pour les images 3D
            image = skimage.transform.resize(image, (299, 299, 299), anti_aliasing = True)
            transformed_image = torch.from_numpy(image).float()
            
            if transformed_image.ndim == 3:
                transformed_image = transformed_image.unsqueeze(0)
                
        else:
            # Traitement pour les images 2D
            if image.ndim == 2:
                image = skimage.color.gray2rgb(image)
                
            image = skimage.transform.resize(image, (299, 299), anti_aliasing = True)
            transformed_image = torch.from_numpy(image).float().permute(2, 0, 1)

        
        
        return transformed_image
    