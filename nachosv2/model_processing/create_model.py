from termcolor import colored
import torchvision.models as models
import torch.nn as nn
import math

# from nachosv2.model_processing.models.cifar10ff import CIFAR10FF
# from nachosv2.model_processing.models.conv3D import Conv3DModel
# from nachosv2.model_processing.models.inceptionv3 import InceptionV3
# from nachosv2.model_processing.models.resnet3D import ResNet3D
# from nachosv2.model_processing.models.resnet18_3D import ResNet18_3D


# This is a dictionary of all possible models to create. There are not pre-trained
# models_dictionary = {
#     "Cifar10FF": CIFAR10FF,
#     "Conv3DModel": Conv3DModel,
#     "InceptionV3": InceptionV3,
#     "ResNet18-3D": ResNet18_3D,
#     "ResNet3D": ResNet3D,
# }

l_models = [
            "Cifar10FF",
            "Conv3DModel",
            "InceptionV3",
            "ResNet18-3D",
            "ResNet50",
            "vit_b_16",
            ]

def get_model(model_name: str,
              number_classes: int,
              number_channels: int,
              target_dimensions: tuple):
    
    new_dimensions = None
    # https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py
    # https://pytorch.org/vision/0.12/generated/torchvision.models.inception_v3.html
    if model_name == "InceptionV3":
        # For the parameter aux_logits, which uses and additional branch
        # to help with gradient flow, useful for vanishing gradient problem
        # If images are less than 299x299, then aux_logits=False
        # If images are at least 75x75, then aux_logits=True
        model = models.inception_v3(weights=None,
                                    init_weights=True,
                                    num_classes=number_classes,
                                    aux_logits=False)
        

        model.Conv2d_1a_3x3.conv = nn.Conv2d(
            in_channels=number_channels,  # Change to 1 channel for grayscale
            out_channels=model.Conv2d_1a_3x3.conv.out_channels,
            kernel_size=model.Conv2d_1a_3x3.conv.kernel_size,
            stride=model.Conv2d_1a_3x3.conv.stride,
            padding=model.Conv2d_1a_3x3.conv.padding,
            bias=model.Conv2d_1a_3x3.conv.bias is not None
        )
        
    elif model_name == "ResNet50":
        # ResNet50-specific logic
        model = models.resnet50(weights=None,
                                num_classes=number_classes)
        # Adjust the first convolutional layer to handle a different number of input channels
        model.conv1 = nn.Conv2d(
            in_channels=number_channels,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias is not None
        )
    elif model_name == "vit_b_16":
        # Ensure that the target dimensions are divisible by 16        
        n_patches = max(math.floor(target_dimensions[0]/16.0),
                        math.floor(target_dimensions[1]/16.0))
        
        new_dimensions = (n_patches * 16, n_patches * 16)
        
        print("New dimensions for ViT model:", new_dimensions)
        
        # Adjusting image to match patch size 
        if len(target_dimensions) != 2:
            raise ValueError(colored(f"Error: Target dimensions for ViT model should be a tuple of two integers, got {target_dimensions}.", 'red'))

        # Source code: https://docs.pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html#vit_b_16
        model = models.vit_b_16(weights=None,
                                image_size=new_dimensions[0],
                                num_classes=number_classes)
        
        # Modify the first patch embedding layer to accept 1-channel input
        # Original: Conv2d(3, embed_dim, kernel_size=16, stride=16)
        old_conv = model.conv_proj
        model.conv_proj = nn.Conv2d(
            in_channels=number_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
    
    elif model_name == "vit_b_32":
        model = models.vit_b_32(weights=None,
                                num_classes=number_classes)

        # Source code: https://docs.pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html#vit_b_16
        # Modify the first patch embedding layer to accept 1-channel input
        # Original: Conv2d(3, embed_dim, kernel_size=16, stride=16)
        old_conv = model.conv_proj
        model.conv_proj = nn.Conv2d(
            in_channels=number_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
    else:
        raise ValueError(colored(f"Error: Model '{model_name}' not found in the list of possible models: {l_models}.", 'red'))

    return model, new_dimensions


def create_model(configuration_file:dict,
                 hyperparameters:dict):
    """
    Creates and prepares a model for training.
        
    Args:
        model_type (str): Name of the type of model to create.
        class_names (list of str): List of all classes. Use to know how many there are.
        
    Returns:
        model (nn.Module): The prepared torch.nn model.
    """

        
    # Sets the model definition
    model_name = hyperparameters["architecture"]
    number_classes = len(configuration_file["class_names"])
    number_channels = configuration_file["number_channels"]
    
    if model_name not in l_models:
        raise ValueError(colored(f"Error: Model '{model_name}' not found in the list of possible models: {l_models}.", 'red'))

    # # Gets the model
    # ModelClass = models_dictionary[model_type]
    
    # # Creates the model
    # training_model = ModelClass(configuration_file)

    training_model, new_target_dimensions = get_model(
        model_name=model_name,
        number_classes=number_classes,
        number_channels=number_channels,
        target_dimensions=configuration_file["target_dimensions"])

    return training_model, new_target_dimensions