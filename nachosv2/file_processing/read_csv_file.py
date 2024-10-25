import os
from termcolor import colored


def read_csv_file(file_path):
    """
    Reads the first line of a CSV file and returns it as a list of strings.

    Args:
        file_path (str): Path to the CSV file to read.
    
    Returns:
        values (list of str): The list of strings containing the values of the file.
    """
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            values = file.readline().split(',')
            
            return values
    
    
    except FileNotFoundError:
        print(colored(f"The file '{file_path}' was not found.", 'red'))
    
    except Exception as error:
        print(colored(f"An error occurred while opening '{file_path}': {error}", 'red'))
    