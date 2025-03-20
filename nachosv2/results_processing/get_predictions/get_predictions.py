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
from nachosv2.results_processing.get_metrics.get_metrics import (
    generate_metrics_file
)


def main():
    """
    The main body of the program
    """
    # Parses the command line arguments
    args = parse_command_line_args()
    # Defines the arguments
    config_dict = get_config(args['file'])

    results_path = Path(config_dict['results_path'])
    ensure_path_exists(results_path)

    output_path = config_dict.get('output_path', None)
    if output_path is not None:
        output_path = Path(output_path)

    is_cv_loop = config_dict.get('is_cv_loop', None)


if __name__ == "__main__":
    main()
