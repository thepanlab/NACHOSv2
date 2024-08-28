import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from src.results_processing.learning_curve import learning_curve
from src.file_processing import path_getter
from src.results_processing.results_processing_utils.get_configuration_file import parse_json



def run_program(args):
    """
    Runs the program for each item.

    Args:
        args (dict): The configuration arguements.
    """
    
    # Gets the needed input paths, as well as the proper file names for output
    subfold_paths = path_getter.get_subfolds(args["data_path"])
    
    json = {
        label: args[label] for label in (
            'training_loss_line_color', 'validation_loss_line_color',
            'training_accuracy_line_color', 'validation_accuracy_line_color',
            'font_family', 'label_font_size', 'title_font_size',
            'save_resolution', 'save_format', 'output_path'
        )
    }

    # For each item, run the program
    for model in subfold_paths:
        for subject in subfold_paths[model]:
            for subfold in subfold_paths[model][subject]:
                
                # Get the program's arguments and run
                json['input_path'] = subfold
                learning_curve.main(json)



def main():
    """
    The Main Program.
    """
    
    # Gets program configuration and run using its contents
    config = parse_json(os.path.join(os.path.dirname(__file__), 'learning_curve_many_config.json'))
    run_program(config)



if __name__ == "__main__":
    """
    Executes Program.
    """
    
    main()
    