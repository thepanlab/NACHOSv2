"""get_learning_curve.py
"""
from pathlib import Path
from typing import Optional
from termcolor import colored
import pandas as pd
from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.utils import get_filepath_list
from nachosv2.setup.utils import get_filepath_from_results_path
from nachosv2.setup.files_check import ensure_path_exists
from nachosv2.setup.get_config import get_config
from nachosv2.setup.utils_processing import save_dict_to_yaml
from nachosv2.setup.utils_processing import parse_filename
from nachosv2.setup.utils_processing import is_metric_allowed
from nachosv2.setup.utils import get_new_filepath_from_suffix
import matplotlib.pyplot as plt


def generate_individual_history(history_path: Path,
                                is_cv_loop: bool,
                                output_filepath: Path,
                                title: Optional[str]=None):
    df_history = pd.read_csv(history_path,
                             index_col=0)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10,5)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    
    if not title:
        ax.set_title("Model Loss Over Epochs")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.plot(df_history["training_loss"],
            label="Training")
    if is_cv_loop:
        ax.plot(df_history["validation_loss"],
                label="Validation")
    ax.legend()
    ax.grid(which = "major", axis = "y")
    
    fig.savefig(output_filepath,
                bbox_inches="tight",
                dpi=300)


def generate_learning_curve(
        results_path: Path,
        is_cv_loop: bool,
        custom_output_dir: Optional[Path]=None):

    suffix_filename = "history"

    history_path_list = get_filepath_list(results_path,
                                          suffix_filename,
                                          is_cv_loop)

    for history_path in history_path_list:

        learning_curve_path = get_new_filepath_from_suffix(
                                input_filepath=history_path,
                                old_suffix="history",
                                new_suffix="learning_curve",
                                is_cv_loop=is_cv_loop,
                                custom_output_dir=custom_output_dir)
        
        learning_curve_path = learning_curve_path.with_suffix(".png")

        generate_individual_history(history_path,
                                    is_cv_loop,
                                    learning_curve_path)


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
    results_path = Path(config_dict['results_path'])
    ensure_path_exists(results_path)

    output_path = config_dict.get('output_path', None)
    if output_path is not None:
        output_path = Path(output_path)

    is_cv_loop = config_dict.get('is_cv_loop', None)
     
    # ------------------------------
    # Step 3: Generate learning curves
    # ------------------------------

    generate_learning_curve(
        results_path=results_path,
        is_cv_loop=is_cv_loop,
        custom_output_dir=output_path)


if __name__ == "__main__":   
    main()
    