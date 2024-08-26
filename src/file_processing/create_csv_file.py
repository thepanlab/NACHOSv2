from termcolor import colored
import csv
import os


def create_empty_csv(file_path, headers):
    """
    Creates an empty CSV file with the provided headers.

    Args:
        file_path (str): Name of the CSV file to create.
        headers (list of str): List of column names.
    """
    
    try:
        # Ensures the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        
        # Opens the file in write mode
        with open(file_path, mode = 'w', newline = '', encoding = 'utf-8') as csv_file:
            
            # Creates a writer object
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            
            # Writes the headers
            writer.writeheader()
        
        
    except IOError as e:
        print(colored(f"An error occurred while creating the file: {e}", 'red'))
        
        
    except Exception as e:
        print(colored(f"An unexpected error occurred: {e}", 'red'))
        