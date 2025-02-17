from typing import List, Union
from termcolor import colored


def get_list_of_epochs(epoch_values: Union[int, list],
                       test_subjects: List[str],
                       is_cv_loop: bool = False,
                       is_verbose_on: bool = False):
    """
    Returns list of epochs. If unique value is given it repeats
    according to the length of test_subjects list. Otherwise,
    it returns lists of epochs for each subject test

    Args:
        epoch_values (int or list): The number of epochs or the list of number of epochs.
        test_subjects (list): The list of test subjects.
        is_outer_loop (bool): If this is of the outer loop.
        is_verbose_on (bool): If the verbose mode is activated. Default is false. (Optional)
    
    Returns:
        list_of_epochs_for_each_subject (list): The list of epochs for each subjects
        
    Raises:
        ValueError: if epochs_values is a list when inner loop
        ValueError: if len of epochs != len test subjects
    """
    
    if isinstance(epoch_values, int):
        # Single integer provided, create a list repeating the value for each test subject
        return [epoch_values] * len(test_subjects)

    elif isinstance(epoch_values, list):
        # Handle list of epoch values
        if len(epoch_values) == 1 and is_cv_loop:
            # Single value in the list, use it for all test subjects (inner loop)
            return [epoch_values[0]] * len(test_subjects)

        if len(epoch_values) > 1 and is_cv_loop:
            # Multiple values in the list for the inner loop is not allowed
            raise ValueError(colored("For inner loop, you should have only one value for epochs.", 'red'))

        if len(epoch_values) != len(test_subjects):
            # Ensure the length of epoch_values matches the length of test_subjects
            raise ValueError(colored(
                f"Length of list of epochs ({len(epoch_values)}) does not match the length of test_subjects ({len(test_subjects)}).",
                'red'
            ))

        # Valid list of epoch values provided
        return epoch_values
    
    else:
    # Raise error if epoch_values is neither an int nor a list
        raise TypeError(colored("hyperparameters:epochs must be an int or a list.", 'red'))
