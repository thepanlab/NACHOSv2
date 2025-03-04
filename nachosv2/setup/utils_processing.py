from pathlib import Path
import yaml

def save_dict_to_yaml(data_dict: dict,
                      file_path: Path):
    """
    Save a dictionary to a YAML file.
    
    Args:
        data_dict (dict): The dictionary to save.
        file_path (Path): The file path where the YAML file will be saved.
        
    Returns:
        None
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data_dict, file, sort_keys=False)
        print(f"Dictionary saved successfully to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")