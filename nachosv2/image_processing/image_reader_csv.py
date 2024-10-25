import numpy as np

from src.image_processing.image_utils import *
from src.image_processing.image_reader import ImageReader
from src.file_processing.file_name_utils import get_file_name, remove_file_extension
from src.image_processing.image_crop import crop_image


class ImageReaderCSV(ImageReader):
    """
    Handles reading and parsing of .csv images.

    CSV images are special as they cannot be read by io.imread().
    Stored as 1-Dimensional arrays, when they are loaded, they will need to be reshaped.
    """
    
    def __init__(self, configs):
        self._configs = configs
        ImageReader.__init__(self)
        return



    def io_read(self, filename):
        image = np.genfromtxt(filename)
        if "csv_shape" in self._configs:
            image = image.reshape(self._configs["csv_shape"])
        else:
            image = image.reshape(185, 210, 185, 1)

        return image



    def parse_image(self, filename, class_names, channels, do_cropping, offset_height, offset_width, target_height, target_width, label_position=None, use_labels=True): 

        # Split to get only the image name
        image_path = get_file_name(filename)
        
        # Remove the file extention
        path_substring = remove_file_extension(image_path)
        
        # Find the label
        label = extract_label_from_path(path_substring, "_", label_position)

        label_bool = get_label_from_class_names(label, class_names)

        image = self.io_read(filename)

        # Crop the image
        if do_cropping == 'true':
            image = crop_image(image, (offset_height, offset_width, target_height, target_width))
        
        return image, label_bool
    