from termcolor import colored


def get_indexes(image_path_list, class_names, subject_list, is_verbose_on = False):
    """
    Gets the indexes of classes, and the subjects within the image file names.
        
    Args:
        image_path_list (list of str): The list of input image paths.
        class_names (list of str): The list of class names.
        subject_list (list of str): The list of subject names.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
        
    Returns:
        indexes (dict of lists): A dictionary containing all labels, indexes, and subjects.
        label_position (int): The label position.
        
    Exception:
        When more than one label or subject is given.
    """
    
    # Initialization
    indexes = {'labels': [], 'indexes': [], 'subjects': []}

    
    # For each file in the list
    for file in image_path_list:
        
        # Gets the proper filename to parse
        formatted_name = file.replace("%20", " ").split('/')[-1].split('.')[:-1] # Replaces'%20' with space, splits by '/' and taking the last part, splits it by '.' and removes the last element (file extension)
        formatted_name = '.'.join(formatted_name) # Joins the formatted name parts with '.' to form a string without the file extension
        formatted_name = formatted_name.split('_') # Splits the formatted name by '_' to further process it
        
        
        # Gets the image label
        labels = [c for c in class_names if c in formatted_name] # Filters class names to keep only those present in the formatted name
        
        # Checks if there is exactly one label found
        if len(labels) != 1: # Raises an error if not
            raise ValueError(colored(f"Error: {len(labels)} labels found for '{file}'.\n"
                                     "There should be exactly one. Is the case correct?",
                                     'red'))
        
        
        # Gets the index of the label in the class name list
        label_index = class_names.index(labels[0])
        
            
        # Gets the image subject
        subjects = [s for s in subject_list if s in formatted_name]
        if len(subjects) != 1:
            raise Exception(colored(f"Error: {len(subjects)} subjects found for '{file}'.\n"
                                    "There should be exactly one. Is the case correct?",
                                    'red'))
        
        
        # Gets the position of the label in the string
        label_position = formatted_name.index(labels[0])

            
        # Adds the values to the dictionary
        indexes['labels'].append(labels[0])
        indexes['indexes'].append(label_index)
        indexes['subjects'].append(subjects[0])
        
        
    if is_verbose_on: # If the verbose mode is activated
        print(colored('Finished finding the label indexes.', 'cyan'))
        
        
    return indexes, label_position
 