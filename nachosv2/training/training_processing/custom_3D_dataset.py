import numpy as np
from skimage import io

from torch.utils.data import Dataset

# from nachosv2.image_processing.image_transformations import image_transformation_3D


# class Custom3DDataset(Dataset):
#     def __init__(self, prefix, data_dictionary):
#         """
#         Initializes a custom dataset object.
        
#         Args:
#             prefix (str): The file path prefix = the path to the directory where the images are.
#             data_dictionary (dict of list): The data dictionary.
#         """
        
#         self.prefix = prefix
#         self.data_dictionary = data_dictionary
#         self.file_list = data_dictionary['files']



#     def __len__(self):
#         """
#         Returns the dataset len.
#         """
        
#         return len(self.file_list)



#     def __getitem__(self, index):
#         """
#         Retrieves a single sample from the dataset given an index.
        
#         Args:
#             index (int): The index of a sample in the dataset.
        
#         Returns:
#             image_tensor (PyTorch tensor): The sample from the dataset.
#             label_index (int): The label of the sample.
#             image_index (int): The index of the sample in all the dataset.
#             file_path (str): The file path.
#         """
        
#         # Gets the image path
#         image_path = f"{self.prefix}/{self.file_list[index]}"
        
#         # Reads the image
#         image = io.imread(image_path)
        
#         # Adds channel dimension
#         image = np.expand_dims(image, axis = 0)  
        
#         # Transformes the images
#         image = image_transformation_3D(image)
            
        
#         # Gets the label
#         label_index = self.data_dictionary['labels'][index]
        
#         # Gets the index
#         image_index = self.data_dictionary['indexes'][index]
        
#         # Gets the file path
#         file_path = self.file_list[index]
        
        
#         return image, label_index, image_index, file_path