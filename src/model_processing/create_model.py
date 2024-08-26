from termcolor import colored
import sys

from src.model_processing.models.cifar10ff import CIFAR10FF
from src.model_processing.models.conv3D import Conv3DModel
from src.model_processing.models.inceptionv3 import InceptionV3
from src.model_processing.models.simple3DCNN import Simple3DCNN
from src.model_processing.models.resnet3D import ResNet3D
from src.model_processing.models.resnet18_3D import ResNet18_3D


# This is a dictionary of all possible models to create. There are not pre-trained
models_dictionary = {
    "Cifar10FF": CIFAR10FF,
    "Conv3DModel": Conv3DModel,
    "InceptionV3": InceptionV3,
    "ResNet18-3D": ResNet18_3D,
    "ResNet3D": ResNet3D,
    "Simple3DCNN": Simple3DCNN,
}



def create_training_model(configuration_file):
    """
    Creates and prepares a model for training.
        
    Args:
        model_type (str): Name of the type of model to create.
        class_names (list of str): List of all classes. Use to know how many there are.
        
    Returns:
        model (nn.Module): The prepared torch.nn model.
    """

    try:            
        # Sets the model definition
        model_type = configuration_file["selected_model_name"]

        # Gets the model
        ModelClass = models_dictionary[model_type]
        
        # Creates the model
        training_model = ModelClass(configuration_file)
        

        return training_model

            
    # If the model type is not in the model list, error
    except KeyError:
        print(colored(f"Error: Model '{model_type}' not found in the list of possible models: {list(models_dictionary.keys())}.", 'red'))
        sys.exit()
