from pathlib import Path

import pandas as pd

from nachosv2.setup.get_config import get_config

                                
def verify_single_values(hyperparameter: dict):
    for key, value in hyperparameter.items():
        if isinstance(value, dict):
            raise ValueError(f"Hyperparameter {key} should have subfields (e.g. min, max). It must be a single value or a list of one value.")
        elif not isinstance(value, int) or \
            (isinstance(value, list) and len(value)>1):
            raise ValueError(f"Hyperparameter {key} must be a single value or a list of one value.")       


def extract_values(hyperparameter: dict):
    dict_values = {}
    
    for key, value in hyperparameter.items():
        if isinstance(value, list):
            dict_values[key] = value[0]
        else:
            dict_values[key] = value
    
    return dict_values


def create_random_configurations(hyperparameter_dict):
    
    default_filename = "hpo_default_values.csv"
    current_file_path = Path(__file__).resolve()
    dir_path = current_file_path.parent
    
    # Path to the CSV file
    default_path = dir_path / default_filename
    
    # Check if the file exists and is a file
    if not default_path.exists() or not default_path.is_file():
        raise FileNotFoundError(f"The file {default_path} does not exist or is not a file")
    
    df_default = pd.read_csv(default_path, index_col=0)
    
    # Go through the default values and check if they are in the hyperparameter_dict
    
    n_combinations = dict_hyperparameters_default["n_combinations"]
    
    l_dict = []
    for i in range(n_combinations):
        dict_temp = {}
        for key, value in dict_hyperparameters_default.items():
            if key not in hyperparameter_dict:
                dict_temp[key] = value
            
                


def get_hpo_configuration(config):
    
    hyperparameter_dict = get_config(config["configuration_filepath"])
    
    if not config["use_hpo"]:
        # Verify there are single values or list of one value
        verify_single_values(hyperparameter_dict)
        hyperparameter_dict = extract_values(hyperparameter_dict)
    else:
        create_random_configurations(hyperparameter_dict)
