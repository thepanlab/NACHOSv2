"""get_learning_curve.py
"""
from pathlib import Path
from typing import Optional, List
from termcolor import colored
from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.files_check import ensure_path_exists
from nachosv2.setup.get_config import get_config
from nachosv2.model_processing.create_model import get_model
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from nachosv2.output_processing.result_outputter import save_dict_or_listdict_to_csv


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
                         inputs,
                         device):
    with torch.set_grad_enabled(False):
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        # Gets the predicted class indices
        _, class_predictions = torch.max(outputs, 1)
    # print(i, filepath, class_predictions.item(), labels.item())
    
    return class_predictions.item(), probabilities.cpu().numpy()


def predict_single_image(model, config_dict,
                         image_path, transform,
                         device):
    if not image_path.exists():
        raise FileNotFoundError(f"Image path {image_path} does not exist.")

    if config_dict["number_channels"] == 1:
        image = Image.open(image_path).convert("L")
    elif config_dict["number_channels"] == 3:
        image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
    return process_single_image(model,
                                input_tensor,
                                device)
    

def get_probabilities_one_image(model,
                                config_dict,
                                image_path,
                                transform,
                                class_names,
                                device):
    dict_temp = {}
    
    prediction, probabilities = predict_single_image(
        model,
        config_dict,
        Path(image_path),
        transform,
        device)

    dict_temp["image_path"]=image_path
    dict_temp["prediction"]=prediction
        
    for i, class_name in enumerate(class_names):
        dict_temp[f"probability_{i}_{class_name}"] = probabilities[0][i]
    
    return dict_temp


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

    class_names = config_dict["class_names"]
    n_classes = len(config_dict["class_names"])

    device = config_dict["device"]

    model = get_model(model_name,
                      number_classes=n_classes,
                      number_channels=config_dict["number_channels"])
    model.to(device)
    model.eval()

    output_folder = Path(config_dict["output_folder"])
    suffix = config_dict.get("suffix", None)

    filename = "predictions.csv"
    if suffix:
        filename = f"{suffix}_{filename}"
    
    validate_exclusive_options(config_dict,
                               ["image_path", "image_folder"])

    if config_dict.get("do_normalize", None):
        transform = transforms.Compose([
            transforms.Resize((config_dict["target_height"],
                               config_dict["target_width"])), # Resize to expected input size
            transforms.ToTensor(),                 # Convert to tensor and normalize to [0, 1]
            transforms.Normalize(                  # Apply same normalization used during training
                mean=config_dict["normalization_values"]["mean"],        # For ImageNet-pretrained models
                std=config_dict["normalization_values"]["stddev"]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config_dict["target_height"],
                               config_dict["target_width"])), # Resize to expected input size
            transforms.ToTensor(),                 # Convert to tensor and normalize to [0, 1]
        ])

    if config_dict.get("image_path"):
        image_path = config_dict.get("image_path")
        dict_temp = get_probabilities_one_image(
                model,
                config_dict,
                image_path,
                transform,
                class_names,
                device)
        
        prediction_rows = [dict_temp]
        
    elif config_dict.get("image_folder"):
        image_folder = Path(config_dict.get("image_folder"))
        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder {image_folder} does not exist.")

        # Get all image files in the folder
        image_files = list(image_folder.glob("*.png")) + \
                      list(image_folder.glob("*.jpg")) + \
                      list(image_folder.glob("*.jpeg"))

        prediction_rows = []

        for image_path in tqdm(image_files):

            dict_temp = get_probabilities_one_image(
                model,
                config_dict,
                image_path,
                transform,
                class_names,
                device)

            prediction_rows.append(dict_temp)
            
    save_dict_or_listdict_to_csv(prediction_rows,
                                 output_folder,
                                 filename)


if __name__ == "__main__":
    main()
