import os
import sys
import pandas as pd
from termcolor import colored
from pathlib import Path

def read_metadata_csv(path_label_csv):
    """
    Reads all CSV files in the given directory and returns a dictionary of the datas.
    
    Args:
        dierctory_path (str): The directory where are the CSV files.
        configuration_file (JSON): The configuration file.
    
    Returns:
        data_dict (dict): A dictionary of of the datas from the CSV files.
                        data_dict = {'subject1': {'image_path': [image_path1, image_path2, ...],
                                                  'indexes': [0, 1, 2, ...],
                                                  'label1_number': [...],
                                                  'label2_number': [...],
                                                  ...,
                                                  },
                                    'subject2': {...},
                                    ...
                                    }
    """
    
    # Initializations
    p = Path(path_label_csv)
    
    if not p.is_file():
        raise FileNotFoundError(f"Error: The file '{p.name}' does not exist in the directory '{p.parent}'.")
    
    df_labels = pd.read_csv(p)
    
    # Checks if the given path is valid

    return df_labels


def read_data_csv(directory_path, configuration_file):
    """
    Reads all CSV files in the given directory and returns a dictionary of the datas.
    
    Args:
        dierctory_path (str): The directory where are the CSV files.
        configuration_file (JSON): The configuration file.
    
    Returns:
        data_dict (dict): A dictionary of of the datas from the CSV files.
                        data_dict = {'subject1': {'image_path': [image_path1, image_path2, ...],
                                                  'indexes': [0, 1, 2, ...],
                                                  'label1_number': [...],
                                                  'label2_number': [...],
                                                  ...,
                                                  },
                                    'subject2': {...},
                                    ...
                                    }
    """
    
    # Initializations
    data_dict = {}   # The dictionary where the data will be put
    
    # Checks if the given path is valid
    try:
        assert os.path.isdir(directory_path), f"Error: '{directory_path}' is not a valid input path."
    
    except AssertionError as error:
        print(colored(error, 'red'))
        sys.exit()

    # Creates the list to read, containing only one time each used subject
    csv_file_to_read_list = set(configuration_file['subject_list']
                                + configuration_file['test_subjects']
                                + configuration_file['validation_subjects'])

    # Loops on each CSV file to read it
    for file_to_read in csv_file_to_read_list:
        
        try:
            # Creates the file name
            file_name = f"{file_to_read}.csv"
            
            # Creates the file path
            file_path = os.path.join(directory_path, file_name)
            
            # Tries to read the file
            with open(file_path, 'r') as f:

                data_dict[file_to_read] = _loop_read_data_csv(file_path)
            
        except FileNotFoundError:
            print(colored(f"Error: The file '{file_name}' does not exist in the directory '{directory_path}'.", 'red'))
            sys.exit()\
    
    return data_dict

def _loop_read_data_csv(file_path):
    """
    Reads the given CSV file and puts it into the dictionary.
    
    Args:
        file_path (str): The path to the CSV file to read.
    """
    
    # Initialization
    subject_data_dict = {}
    
    # Reads the file and puts it into a panda dataframe
    file_dataframe = pd.read_csv(file_path)

    # Saves the image names (first column of hte CSV) into a dictionary
    if not file_dataframe.empty:  # Checks the dataframe is not empty
        
        # Creates the list of image_path
        image_name_list = file_dataframe.iloc[:, 0].tolist()

        # Adds the list to the dictionary
        subject_data_dict['image_path'] = image_name_list

    # Adds the indexes to the dataframe
    subject_data_dict['indexes'] = file_dataframe.index.tolist()
    
    # Adds the data from the dataframe to the dictionary
    for column in file_dataframe.columns[1:]: # Except the first column
        subject_data_dict[column] = file_dataframe[column].tolist()
    
    return subject_data_dict