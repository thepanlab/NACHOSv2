"""get_learning_curve.py
"""
from pathlib import Path
from typing import Optional, List
from termcolor import colored
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from nachosv2.setup.command_line_parser import parse_command_line_args
from pytorch_grad_cam.utils.image import show_cam_on_image
from nachosv2.data_processing.read_metadata_csv import read_metadata_csv
from nachosv2.setup.utils import get_filepath_list
from nachosv2.setup.utils import get_filepath_from_results_path
from nachosv2.setup.files_check import ensure_path_exists
from nachosv2.setup.get_config import get_config
from nachosv2.setup.utils_processing import save_dict_to_yaml
from nachosv2.setup.utils_processing import parse_filename
from nachosv2.setup.utils_processing import is_metric_allowed
from nachosv2.setup.utils_training import get_files_labels_for_fold
from nachosv2.setup.utils import get_new_filepath_from_suffix
from nachosv2.model_processing.create_model import get_model
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from nachosv2.training.training_processing.custom_2D_dataset import Dataset2D


def validate_exclusive_options(config: dict,
                               keys: List[str]) -> None:
    active_keys = [key for key in keys if config.get(key)]
    if len(active_keys) != 1:
        raise ValueError(f"Exactly one of {keys} must be set. Found: {active_keys}")


def get_submodule_by_name(model, name: str):
    submodule = model
    for attr in name.split('.'):
        if attr.isdigit():
            submodule = submodule[int(attr)]
        else:
            submodule = getattr(submodule, attr)
    return submodule


def process_single_image(model,
                         config_dict,
                         inputs,
                         filepath):
    inputs = inputs.to("cuda")
    outputs = model(inputs)
    # Gets the predicted class indices
    _, class_predictions = torch.max(outputs, 1)
    # print(i, filepath, class_predictions.item(), labels.item())

    target_layers = [get_submodule_by_name(model,
                     config_dict["target_layer"])]

    index_class = config_dict.get("class_index_for_explainability", -1)
    index_class =  class_predictions.item() if index_class == -1 else index_class

    targets = [ClassifierOutputTarget(index_class)]

    image = Image.open(filepath)
    image_normalized = np.array(image)/255

    # Construct the CAM object once, and then re-use it on many images.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=inputs, targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(image_normalized,
                                          grayscale_cam,
                                          use_rgb=True)
        # You can also get the model outputs without having to redo inference
        model_outputs = cam.outputs
        
    image = Image.fromarray(visualization)
    
    filename = Path(filepath).stem
    if config_dict.get("suffix", None):
        filename = filename + "_" + config_dict["suffix"]
        
    path_output_folder = Path(config_dict['output_folder'])
    path_output_folder.mkdir(parents=True, exist_ok=True)
        
    filepath_output = path_output_folder / f"{filename}.png"
    
    image.save(filepath_output)
    
    return class_predictions.item()


def apply_gradcam_single_image(model, config_dict, image_path):

    if not image_path.exists():
        raise FileNotFoundError(f"Image path {image_path} does not exist.")

    if config_dict["number_channels"] == 1:
        image = Image.open(image_path).convert("L")
    elif config_dict["number_channels"] == 3:
        image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)    
    prediction = process_single_image(model, config_dict,
                                      input_tensor, image_path)
    print(image_path, prediction)


def main():
    """
    Plots the learning curves for the given file.

    Args:
        config (dict, optional): A custom configuration. Defaults to None.
    """
    # ------------------------------
    # Step 1: Load and parse arguments
    # ------------------------------
    args = parse_command_line_args()
    config_dict = get_config(args['file'])

    # ------------------------------
    # Step 2: Resolve paths and flags
    # ------------------------------

    model_path = Path(config_dict['model_path'])
    ensure_path_exists(model_path)
    model_name = config_dict['model_name']

    n_classes = len(config_dict["class_names"])

    model = get_model(model_name,
                      number_classes=n_classes,
                      number_channels=config_dict["number_channels"])
    model.to("cuda")

    validate_exclusive_options(config_dict,
                               ["fold", "image_path","image_folder"])

    df_metadata = read_metadata_csv(config_dict["metadata_path"])


    if config_dict.get("do_normalize", None):
        transform = transforms.Normalize(mean=config_dict["normalization_values"]["mean"],
                                            std=config_dict["normalization_values"]["stddev"])
    else:
        transform = None

    if config_dict.get("fold"):
        fold_info_dict = {'files': [], 'labels': [], 'dataloader': None}
        fold_info_dict['files'], fold_info_dict['labels'] = get_files_labels_for_fold(
            df_metadata=df_metadata,
            fold_or_list_fold="k1")

        dataset = Dataset2D(
            dictionary_partition=fold_info_dict,
            number_channels=config_dict["number_channels"],
            image_size=(config_dict["target_height"],
                          config_dict["target_width"]),
            do_cropping=False,
            crop_box=None,
            transform=transform
            )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        for i, (inputs, labels, filepath) in enumerate(dataloader):
            prediction = process_single_image(model, config_dict,
                                              inputs, filepath[0])
            print(i, filepath, "truth", labels.item(), "prediction", prediction )
            
    elif config_dict.get("image_path"):
        image_path = config_dict.get("image_path")
        apply_gradcam_single_image(model, config_dict, image_path)
        
    elif config_dict.get("image_folder"):
        
        image_folder = config_dict.get("image_folder")
        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder {image_folder} does not exist.")

        # Get all image files in the folder
        image_files = list(image_folder.glob("*.png")) + \
                      list(image_folder.glob("*.jpg")) + \
                      list(image_folder.glob("*.jpeg"))

        for image_path in image_files:
            apply_gradcam_single_image(model, config_dict, image_path)


if __name__ == "__main__":   
    main()
