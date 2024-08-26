from termcolor import colored


def get_list_of_epochs(epochs_values, test_subjects, is_outer_loop, is_verbose_on = False):
    """
    Returns list of epochs. If unique value is given it repeats
    according to the length of test_subjects list. Otherwise,
    it returns lists of epochs for each subject test

    Args:
        epochs_values (int or list): The number of epochs or the list of numbers of epochs.
        test_subjects (list): The list of test subjects.
        is_outer_loop (bool): If this is of the outer loop.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
    
    Returns:
        list_of_epochs_for_each_subject (list): The list of epochs for each subjects
        
    Raises:
        ValueError: if epochs_values is a list when inner loop
        ValueError: if len of epochs != len test subjects
    """
    
    # Initialization
    is_single_epoch = True # Assumes by default that there is a single epoch value
    
    
    # If epochs_values is an int
    if isinstance(epochs_values, int):
        epochs = epochs_values
    
    
    # If epochs_values is a list
    elif isinstance(epochs_values, list):
        
        # If the list contains only 1 number of epochs and inner loop, extracts the number of epochs
        if len(epochs_values) == 1 and is_outer_loop == False:
            epochs = epochs_values[0]
            
        # If the list contains more than 1 number of epochs and inner loop, raises ValueError
        elif len(epochs_values) > 1 and is_outer_loop == False:
            raise ValueError(colored("For inner loop, you should have only one value for epoch", 'red'))

        # Checks that the list of epochs is the same length as the list of subjects
        if len(epochs_values) != len(test_subjects):
            raise ValueError(colored(f"Length of list of epochs is :{len(epochs_values)},"+
                                f"length of test_subjects is {len(test_subjects)}",
                                'red'))
            
        else: # If epochs_values is a list and passed all the tests
            is_single_epoch = False # Saves that it's a list
        
        
    if is_single_epoch: # If it's an int
        list_of_epochs_for_each_subject = [epochs] * len(test_subjects) # Creates a list of the same number of epochs
        
    else: # If it's a list
        list_of_epochs_for_each_subject = epochs_values # Returns it
            
            
    return list_of_epochs_for_each_subject
