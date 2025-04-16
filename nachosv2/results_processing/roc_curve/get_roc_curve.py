"""get_roc_curve.py
"""
from pathlib import Path
from typing import Optional
from termcolor import colored
import pandas as pd
from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.utils import get_filepath_list
from nachosv2.setup.files_check import ensure_path_exists
from nachosv2.setup.get_config import get_config
from nachosv2.setup.utils import get_new_filepath_from_suffix
from nachosv2.setup.utils import get_other_result
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


def generate_individual_roc_curve(prediction_path: Path,
                                  is_cv_loop: bool,
                                  output_filepath: Path,
                                  title: Optional[str] = None):

    df_class_names = get_other_result(prediction_path, "class_names")
    
    df_class_names.set_index('index', inplace=True)
    series_class_names = df_class_names['class_name']

    df_prediction = pd.read_csv(prediction_path,
                                index_col=0)
    
    # Extract actual and predicted probabilities
    y_true = df_prediction['true_label']
    y_score = df_prediction[[f'class_{i}_prob' for i in range(3)]]

    # Binarize the labels for multiclass ROC
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_true_bin.shape[1]

    # Compute ROC curve and AUC for each class
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score.iloc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f'{series_class_names[i]} (AUC = {roc_auc[i]:.2f})')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if not title:
        ax.set_title('Multi-Class ROC Curve')
    ax.legend()
    ax.grid(True)
    
    fig.savefig(output_filepath,
                bbox_inches="tight",
                dpi=300)
    plt.close(fig)


def generate_roc_curve(
        results_path: Path,
        is_cv_loop: bool,
        custom_output_dir: Optional[Path] = None):

    suffix_filename = "prediction"

    prediction_path_list = get_filepath_list(results_path,
                                             suffix_filename,
                                             is_cv_loop)

    for prediction_path in prediction_path_list:

        roc_curve_path = get_new_filepath_from_suffix(
                                input_filepath=prediction_path,
                                old_suffix="prediction",
                                new_suffix="roc_curve",
                                is_cv_loop=is_cv_loop,
                                custom_output_dir=custom_output_dir)
        
        roc_curve_path = roc_curve_path.with_suffix(".png")

        generate_individual_roc_curve(prediction_path,
                                      is_cv_loop,
                                      roc_curve_path)


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

    generate_roc_curve(
        results_path=results_path,
        is_cv_loop=is_cv_loop,
        custom_output_dir=output_path)


if __name__ == "__main__":   
    main()
