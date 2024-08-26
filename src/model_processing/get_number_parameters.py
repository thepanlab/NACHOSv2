import os
import sys
from termcolor import colored

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
import setup_paths

from src.model_processing.create_model import create_training_model
from src.results_processing.results_processing_utils.get_config import parse_json


def get_number_parameters(model):
    """
    TODO
    """
    
    # Init
    number_of_parameters = 0
    
    for parameter in model.parameters():
        number_of_parameters += parameter.numel()
    
    
    return number_of_parameters



def main():
    """
    TThe main program.
    Gaves the number of parameters of the model in the configuration file
    """
    
    configuration = parse_json(os.path.abspath('scripts/config_files/conv3D.json'))
    
    model = create_training_model(configuration)
    
    number_of_parameters = get_number_parameters(model)
    
    print(colored(f"The model {model.model_type} has {number_of_parameters} parameters.", 'green'))
    


if __name__ == "__main__":
    """
    Executes Program.
    """
    
    main()