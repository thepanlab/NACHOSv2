from pathlib import Path
import pandas as pd
from nachosv2.setup.get_config import get_config

                                
def verify_single_values(hyperparameter: dict):
    
    l_dict_allowed = ["cropping_position"]
    
    for key, value in hyperparameter.items():
        if key in l_dict_allowed:
            continue
        elif isinstance(value, dict):
            raise ValueError(f"Hyperparameter {key} have subfields (e.g. min, max). It must be a single value or a list of one value.")
        elif isinstance(value, list) and len(value)>1:
            raise ValueError(f"Hyperparameter {key} must be a single value or a list of one value.")       


def convert_type(val: str,
                 typ: str):
    if typ == 'int':
        return int(val)
    elif typ == 'bool':
        return bool(val)
    elif typ == 'float':
        return float(val)
    elif typ == 'str':
        return str(val)
    else:
        return val


def extract_values_single(df_default:pd.DataFrame,
                          hyperparameter_dict: dict):
    dict_values = {}
    df_temp = df_default.set_index("hyperparameter")
    df_temp['value_converted'] = df_temp.apply(lambda row: convert_type(row['value'],
                                                              row['type']),
                                               axis=1)
    
    dict_values["hp_config_index"] = 0
    
    for index in df_temp.index:
        if index in hyperparameter_dict:
            dict_values[index] = hyperparameter_dict[index]
        else:
            dict_values[index] = df_temp.loc[index, "value_converted"]
    
    return dict_values


def extract_default_hyperparameters() -> pd.DataFrame:
    default_filename = "hpo_default_values.csv"
    current_file_path = Path(__file__).resolve()
    dir_path = current_file_path.parent
    
    # Path to the CSV file
    default_path = dir_path / default_filename
    
    # Check if the file exists and is a file
    if not default_path.exists() or not default_path.is_file():
        raise FileNotFoundError(f"The file {default_path} does not exist or is not a file")
    
    df_default = pd.read_csv(default_path, index_col=0)
    
    return df_default


def create_random_configurations(is_cv_loop: bool,
                                 hyperparameter_dict: dict):
    
    df_default = extract_default_hyperparameters()
    # Go through the default values and check if they are in the hyperparameter_dict
    n_combinations = hyperparameter_dict["n_combinations"]
    l_dict = []
    if n_combinations == 1:
        verify_single_values(hyperparameter_dict)
        l_dict.append(extract_values_single(df_default, hyperparameter_dict))
    else:
        for i in range(n_combinations):
            dict_temp = {}
            for key, value in dict_hyperparameters_default.items():
                if key not in hyperparameter_dict:
                    dict_temp[key] = value


def get_hpo_configuration(config):
    
    hyperparameter_dict = get_config(config["configuration_filepath"])
    
    df_default = extract_default_hyperparameters()
    
    l_hpo_configuration = []
    if not config["use_hpo"]:
        # Verify there are single values or list of one value
        verify_single_values(hyperparameter_dict)
        l_hpo_configuration.append(extract_values_single(df_default, hyperparameter_dict))
    else:
        # TODO
        create_random_configurations(hyperparameter_dict)

    return l_hpo_configuration