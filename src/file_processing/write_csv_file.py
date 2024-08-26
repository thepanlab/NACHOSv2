import csv
from termcolor import colored


def write_to_csv(file_path, data):
    """
    Writes data to an existing CSV file.

    Args:
        file_path (str): Name of the CSV file to write to.
        data (list of dict): List of dictionaries containing the data.
    """
    
    try:
        # Opens the file in append mode
        with open(file_path, mode = 'a', newline = '', encoding = 'utf-8') as csv_file:
        
            if data:
                
                # Creates a writer object
                writer = csv.DictWriter(csv_file, fieldnames = data[0].keys())
                
                # Writes the data rows
                for current_row in data:
                    writer.writerow(current_row)
    
    
    except FileNotFoundError:
        print(colored(f"The file '{file_path}' was not found.", 'red'))
