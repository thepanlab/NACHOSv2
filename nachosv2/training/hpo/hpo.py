import random
import math
from typing import List, Dict
from pathlib import Path
import pandas as pd
from nachosv2.setup.get_config import get_config
from nachosv2.setup.utils import get_folder_path

                  
def verify_single_values(hyperparameter: dict):
    """
    Verify that the hyperparameters are single values or lists of one value.

    Args:
        hyperparameter (dict): Dictionary containing hyperparameters to verify.

    Raises:
        ValueError: If any hyperparameter is a dictionary (except for allowed keys) or a list with more than one value.
    """
    # only value allowed to be a list with more than one value
    l_dict_allowed = ["cropping_position", "learning_rate_scheduler_parameters"]
    for key, value in hyperparameter.items():
        if key in l_dict_allowed:
            continue
        elif isinstance(value, dict):
            raise ValueError(f"Hyperparameter {key} have subfields (e.g. min, max). It must be a single value or a list of one value.")
        elif isinstance(value, list) and len(value)>1:
            raise ValueError(f"Hyperparameter {key} must be a single value or a list of one value.")       


def convert_type(val: str,
                 type_val: str):
    """
    Convert a string value to a specified type.

    Args:
        val (str): The value to convert.
        typ (str): The type to convert the value to. Supported types are 'int', 'float', 'bool', and 'str'.

    Returns:
        The converted value in the specified type.

    Raises:
        ValueError: If the specified type is not supported.
    """

    converter_dict = {
        'int': int,
        'bool': bool,
        'float': float,
        'str': str
    }
    
    if isinstance(type_val, float) and math.isnan(type_val):
        return None
    
    return converter_dict[type_val](val)


def extract_values_single(df_default:pd.DataFrame,
                          hyperparameter_dict: dict):
    """
    Extract single values for each hyperparameter from the default values.

    Args:
        df_default (dict): Default values for the hyperparameters.
        hyperparameter_dict (dict): Dictionary containing hyperparameter information.

    Returns:
        dict: A dictionary with the extracted single values for each hyperparameter.
    """
    dict_values = {}
    dict_values["hp_config_index"] = 0
    
    for index in df_default.index:
        # if values is in hyperparameter configuration
        if index in hyperparameter_dict:
            dict_values[index] = hyperparameter_dict[index]
        # otherwise use default
        else:
            dict_values[index] = df_default.loc[index, "value_converted"]
    
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
    
    # Set the 'hyperparameter' column as the index
    df_default = df_default.set_index("hyperparameter")
    # Convert the 'value' column to the appropriate type
    df_default['value_converted'] = df_default.apply(lambda row: convert_type(row['value'],
                                                                 row['type']),
                                                     axis=1)
    
    return df_default


def random_power(min_value, max_value, base):
    # Get the exponent range
    min_exponent = math.floor(math.log(min_value, base))
    max_exponent = math.ceil(math.log(max_value, base))
    
    # Generate a random exponent within the range
    random_exponent = random.randint(min_exponent, max_exponent)
    
    # Return 10 raised to the chosen exponent
    return base ** random_exponent


def random_value_factor(min_value, max_value, scale_factor=10):
    """
    Generates a random value by scaling a randomly chosen factor within the given range.

    Parameters:
    - min_value (int): Minimum value of the range.
    - max_value (int): Maximum value of the range.
    - scale_factor (int): The scaling factor (default is 10).

    Returns:
    - int: A randomly generated value based on the scale factor.
    """
    if min_value > max_value:
        raise ValueError("min_val should not be greater than max_val")

    # Compute scaling factors
    min_factor = min_value//scale_factor
    max_factor = max_value//scale_factor
    
    # Generate a random exponent within the range
    random_factor = random.randint(min_factor, max_factor)
    
    # Return 10 raised to the chosen exponent
    return scale_factor*random_factor


def get_value_from_hyperparameter_dict(index: str,
                                       hyperparameter_dict: dict,
                                       df_default):
    """
    Retrieve a value from the hyperparameter dictionary or default DataFrame.

    Args:
        index (str): The key for the hyperparameter to retrieve.
        hyperparameter_dict (dict): Dictionary containing hyperparameters and their ranges or values.
        df_default (pd.DataFrame): DataFrame containing default values for hyperparameters.

    Returns:
        The value of the hyperparameter, either from the hyperparameter dictionary or the default DataFrame.

    Raises:
        ValueError: If the hyperparameter range is invalid.
    """
    
    # specify the random function for each hyperparameter
    random_function = {
        "batch_size": lambda min_val, max_val: random_power(min_val, max_val, base=2),
        "n_epochs": lambda min_val, max_val: random_value_factor(min_val, max_val, scale_factor=10),
        "n_patience": random.randint,
        "learning_rate": lambda min_val, max_val: random_power(min_val, max_val, base=10),
        "momentum": random.uniform
    }
    
    value = hyperparameter_dict.get(index, df_default.loc[index, "value_converted"])

    if isinstance(value, list):
        return random.choice(value) if len(value) > 1 else value[0]
    if isinstance(value, dict):
        if "min" in value and "max" in value:
            return random_function[index](value["min"], value["max"])
        else:
            raise ValueError(f"Hyperparameter {index} must have 'min' and 'max' keys.")
    
    return value
    

def is_repeated(dict_values, l_dict):
    """
    Check if a given dictionary of hyperparameter values is already present in a list of dictionaries.

    Args:
        dict_values (dict): Dictionary containing hyperparameter values to check.
        l_dict (List[dict]): List of dictionaries containing previously generated hyperparameter values.

    Returns:
        bool: True if the dictionary of hyperparameter values is already present in the list, False otherwise.
    """
    for existing_dict in l_dict:
        dict_hp = existing_dict.copy()
        # delete hp_config_index to compare just the hyperparameter values
        del dict_hp["hp_config_index"]

        # comparing dictionaries of hyperparameters    
        if dict_hp == dict_values:
            return True
    
    return False


def get_one_random_combination(df_default: pd.DataFrame,
                               hyperparameter_dict: Dict[str,any],
                               l_dict: List[Dict[str,any]],
                               max_number_repetitions: int) -> Dict[str,any]:
    """
    Generate one unique random combination of hyperparameters.

    Args:
        df_default (pd.DataFrame): DataFrame containing default values for hyperparameters.
        hyperparameter_dict (dict): Dictionary containing hyperparameters and their ranges or values.
        l_dict (List[dict]): List of dictionaries containing previously generated hyperparameter values.
        max_number_repetitions (int): Maximum number of combinations to try before raising an error.

    Returns:
        dict: A dictionary containing a unique combination of hyperparameter values.

    Raises:
        ValueError: If too many repeated configurations are generated.
    """

    n_repetitions = 0

    while n_repetitions <= max_number_repetitions:
        
        # Generate random hyperparameter values
        dict_values = {}
        
        for hyperparameter in df_default.index:
            if hyperparameter in hyperparameter_dict:
                # Get value from hyperparameter dictionary
                dict_values[hyperparameter] = \
                    get_value_from_hyperparameter_dict(
                    hyperparameter,
                    hyperparameter_dict,
                    df_default)
            else:
                # Use default value if not in hyperparameter dictionary
                dict_values[hyperparameter] = df_default.loc[index, "value_converted"]

        # Check if the generated combination is unique
        if not is_repeated(dict_values, l_dict):
            return dict_values
        
        n_repetitions += 1

    # Raise an error if too many repeated configurations are generated
    raise ValueError(
        "Too many repeated configurations. "
        "The number of possible configurations is too small for the given ranges. "
        "Consider reducing the number of attempts or expanding the range of hyperparameters."
    )


def add_hp_to_df(dict_values: dict,
                 df_hp_rs: pd.DataFrame):
    """
    Add a new set of hyperparameter values to the DataFrame.

    Args:
        dict_values (dict): Dictionary containing hyperparameter values to add.
        df_hp_rs (pd.DataFrame): DataFrame containing previously generated hyperparameter values.

    Returns:
        pd.DataFrame: Updated DataFrame with the new set of hyperparameter values added.
    """
    # Create a temporary DataFrame from the dictionary of hyperparameter values
    df_temp = pd.DataFrame(dict_values, index=[0])
    
    # Concatenate the temporary DataFrame with the existing DataFrame and reset the index
    df_hp_rs = pd.concat([df_hp_rs, df_temp], ignore_index=True)
    
    return df_hp_rs


def create_random_configurations(hyperparameter_dict: Dict[str,any],
                                 df_default: pd.DataFrame,
                                 config: Dict[str,any]) -> List[Dict[str,any]]:
    """
    Generate random configurations based on the provided hyperparameters.

    Args:
        hyperparameter_dict (dict): Dictionary containing hyperparameter information, including the number of combinations.
        df_default (dict): Default values for the hyperparameters.
        config (dict): Configuration dictionary containing output path information.

    Returns:
        List[dict]: A list of dictionaries, each representing a random hyperparameter configuration.
    """
    # Number of combinations to generate    
    n_combinations = hyperparameter_dict["n_combinations"]
    
    # DataFrame to store hyperparameter configurations
    df_hp_rs = pd.DataFrame()
    
    # List to store the generated configurations
    l_dict = []
    if n_combinations == 1:
        # Verify single values if only one combination is needed
        verify_single_values(hyperparameter_dict)
        l_dict.append(extract_values_single(df_default, hyperparameter_dict))
    else:
        for i in range(n_combinations):
             # Generate one random combination
            random_combination = get_one_random_combination(df_default,
                                                            hyperparameter_dict,
                                                            l_dict,
                                                            n_combinations)
            # Add an index to the combination
            random_combination = {"hp_config_index": i, **random_combination}

            # Add the combination to the DataFrame
            df_hp_rs = add_hp_to_df(random_combination,
                                    df_hp_rs)
            
            #  Store the DataFrame to a CSV file using the provided config
            hp_folder_path = get_folder_path(Path(config["output_path"]),
                                             "hp_random_search",
                                             True)
            hp_filepath = hp_folder_path / "hp_configurations.csv"
            df_hp_rs.to_csv(hp_filepath)
            
            l_dict.append(random_combination)
    
    return l_dict


def get_hp_configuration(config: Dict[str,any]) -> List[Dict[str,any]]:
    """
    Generate hyperparameter optimization (HPO) configurations based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing file paths and HPO usage flag.

    Returns:
        List[dict]: A list of dictionaries, each representing a hyperparameter configuration.
    """
    # Retrieve hyperparameter configuration from the specified file
    hyperparameter_dict = get_config(config["configuration_filepath"])

    # Extract default hyperparameter values
    df_default = extract_default_hyperparameters()
    
    # List to store the generated HPO configurations

    if not config["use_hpo"]:
        # If HPO is not used, verify single values or list of one value
        verify_single_values(hyperparameter_dict)
        return [extract_values_single(df_default,
                                      hyperparameter_dict)]
    else:
        # If HPO is used, generate random configurations
        return create_random_configurations(
                    hyperparameter_dict,
                    df_default,
                    config)