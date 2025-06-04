from termcolor import colored
import torchvision.models as models
import torch.nn as nn
import math


def get_new_dimensions(target_dimensions: list,
                       patch_size: int) -> tuple:
            """
            Ensure that the target dimensions are divisible by the patch size.
            """
            n_patches = max(math.floor(target_dimensions[0] / patch_size),
                            math.floor(target_dimensions[1] / patch_size))
            return (n_patches * patch_size, n_patches * patch_size)


def modify_first_conv(conv_layer: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Replace the input Conv2D layer to handle a different number of channels."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None,
    )


def get_model(model_name: str,
              number_classes: int,
              number_channels: int,
              target_dimensions: tuple):
    
    vit_patch_sizes = {
        "vit_b_16": 16, # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.vit_b_16
        "vit_l_16": 16, # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_l_16.html#vit-l-16
        "vit_h_14": 14,
        "vit_b_32": 32,
        "vit_l_32": 32,
    }
    
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
        

        model.Conv2d_1a_3x3.conv = modify_first_conv(
            model.Conv2d_1a_3x3.conv,
            number_channels
            )
        
    elif model_name == "ResNet50":
        # ResNet50-specific logic
        model = models.resnet50(weights=None,
                                num_classes=number_classes)
        # Adjust the first convolutional layer to handle a different number of input channels
        model.conv1 = modify_first_conv(model.conv1, number_channels)

    elif model_name in vit_patch_sizes:

        if len(target_dimensions) != 2:
            raise ValueError(colored(
                f"Error: Target dimensions for ViT model should be a tuple of two integers, got {target_dimensions}.",
                'red'))

        patch_size = vit_patch_sizes[model_name]
        new_dimensions = get_new_dimensions(target_dimensions, patch_size)
        print(f"New dimensions for {model_name}: {new_dimensions}")

        # Dynamically get the model
        model_fn = getattr(models, model_name)
        model = model_fn(
            weights=None,
            image_size=new_dimensions[0],
            num_classes=number_classes
            )
        model.conv_proj = modify_first_conv(model.conv_proj, number_channels)
        
    else:
        raise ValueError(colored(f"Error: Model '{model_name}' not found in the list of possible models.", 'red'))

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

    training_model, new_target_dimensions = get_model(
        model_name=model_name,
        number_classes=number_classes,
        number_channels=number_channels,
        target_dimensions=configuration_file["target_dimensions"])

    return training_model, new_target_dimensions