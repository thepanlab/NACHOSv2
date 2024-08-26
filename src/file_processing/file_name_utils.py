import os
import re


def get_file_name(file_path):
    """
    Gets the file name. Works for images too.
    
    Args:
        file_path (str): The file path.
    
    Returns:
        file_name (str): The file name.
    """
    
    # Gets the name
    file_name = os.path.basename(file_path)
    
    return file_name



def remove_file_extension(file_name):
    """
    Uses a regular expression to remove the file extension from an image or csv file.
    
    Args:
        file_name (str): The file name.
    
    Returns:
        clean_file_name (str): The file name without the extension.
    """
    
    # Removes the extension
    clean_file_name = re.sub(r'\.png$|\.jpg$|\.jpeg$|\.bmp$|\.csv$', '', file_name, flags=re.IGNORECASE)
    
    
    return clean_file_name
