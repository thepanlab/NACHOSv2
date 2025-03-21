def create_empty_history(is_cv_loop: bool,
                         metrics_dictionary: dict):
    """
    TODO
    """
    
    # Creates the base history
    if is_cv_loop:
        history = {'training_loss': [], 'validation_loss': [],
                   'training_accuracy': [], 'validation_accuracy': []}
    else:
        history = {'training_loss': [], 'training_accuracy': []}
    
    # Adds the wanted metrics to the history dictionary
    # add_lists_to_history(history, metrics_dictionary)

    return history

def is_image_3D(configuration: dict) -> bool:
    """
    Defines if the images are 2D or 3D.
    
    Args:
        configuration (dict): The JSON configuration file.
        
    Return:
        is_3d (bool): The boolean that say of its 2D or 3D. False for 2D, True for 3D.
    """
    
    try:      
        # Counts the number of dimensions
        number_of_dimensions = len(configuration["image_size"])

        # The number of dimensions should be 2 or 3
        if number_of_dimensions not in [2,3]:
            raise ValueError("The number of image dimensions must be either 2 or 3.")

        # Defines if it's 2D or 3D
        if number_of_dimensions == 2:
            is_3d = False        
        else:
            is_3d = True

        return is_3d
    
    except ValueError as e:
        print(colored(e, 'red'))
        sys.exit()
