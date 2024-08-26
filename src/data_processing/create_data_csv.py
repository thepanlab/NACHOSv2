import os

from src.file_processing.create_csv_file import create_empty_csv
from src.file_processing.get_from_directory import get_subdirectories_from_directory, get_files_from_directory
from src.file_processing.write_csv_file import write_to_csv


def create_data_csv(data_path, configuration_file, csv_headers):
    """
    Creates the CSV file with image name and label for each subjects in the given configuration.
    
    Args:
        data_path (str): The path where to create the CSV file.
        configuration_file (str): The config's path.
        csv_headers (list of str): The list of headers for the CSV file.
    """
    
    # For each subject
    for current_subject in configuration_file['test_subjects']: # configuration_file['test_subjects'] is a list of string subjet name
        
        # Creates the paths
        current_csv_path = os.path.join(data_path, f"{current_subject}.csv") # The CSV path
        current_data_path = f"{configuration_file['data_input_directory']}/{current_subject}" # The data path


        # Creates the CSV file
        create_empty_csv(current_csv_path, csv_headers)
        
        
        # Creates the subdirectories list
        subdirectories_list = get_subdirectories_from_directory(current_data_path)


        # For each subdirectory. Each subdirectory is a different label
        for label_number in range(len(subdirectories_list)):
            
            # Creates the current subdirectory paths
            current_subdirectory_full_path = f"{current_data_path}/{subdirectories_list[label_number]}" # Only use to get the files
            current_subdirectory_path_to_write = f"{current_subject}/{subdirectories_list[label_number]}" # Only write in the CSV file
            
            # Creates the file list
            file_list = get_files_from_directory(current_subdirectory_full_path)
            
            # Creates the list of data dictionaries
            data_dictionary_list = []
            for file_name in file_list:
                data_dictionary = {"file_name": f"{current_subdirectory_path_to_write}/{file_name}", "label_number": label_number}
                data_dictionary_list.append(data_dictionary)


            # Writes the data dictionaries in the subject CSV file 
            write_to_csv(current_csv_path, data_dictionary_list)
