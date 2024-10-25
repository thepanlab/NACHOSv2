from abc import ABC, abstractmethod

class ImageReader(ABC):
    """
    Abstract ImageReader Class
    """
    
    def __init__(self):
        return
    
    
    
    @abstractmethod
    def io_read(self, filename):
        """
        Reads an image from a file and loads it into memory
        """
        
        pass



    @abstractmethod
    def parse_image(self, filename, mean, use_mean, class_names, do_cropping, offset_height, offset_width, target_height, target_width, label_position=None, use_labels=True): 
        pass
