import skimage.io
import skimage.transform

from torch.utils.data import Dataset

from nachosv2.image_processing.image_crop import crop_image
from nachosv2.image_processing.image_transformations import image_transformation_2D


class Custom2DDataset(Dataset):
    def __init__(self, prefix, data_dictionary, channels, do_cropping, crop_box, normalization):
        """
        Initializes a custom dataset object.
        
        Args:
            prefix (str): The file path prefix = the path to the directory where the images are.
            data_dictionary (dict of list): The data dictionary.
            
            channels (int): The number of channels in an image.
            
            do_cropping (bool): Whether to crop the image.
            crop_box (tuple of int): The dimensions after cropping.
            
            normalization (transforms.Compose): The image normalization.
        """
        
        self.prefix = prefix
        self.data_dictionary = data_dictionary
        self.file_list = data_dictionary['files']
        
        self.channels = channels
        
        self.do_cropping = do_cropping
        self.crop_box = crop_box
        
        self.normalization = normalization



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
        image = image_transformation_2D(image)
        
        
        # If asked, crops the image
        if self.do_cropping:
            image = crop_image(image, self.crop_box)

        
        # Converts the image into a tensor
        image_tensor = self.normalization(image)


        # Gets the label
        label_index = self.data_dictionary['labels'][index]
        
        # Gets the index
        image_index = self.data_dictionary['indexes'][index]
        
        # Gets the file path
        file_path = self.file_list[index]

        
        return image_tensor, label_index, image_index, file_path
    