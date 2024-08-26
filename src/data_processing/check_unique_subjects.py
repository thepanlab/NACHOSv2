from termcolor import colored
import sys


def check_unique_subjects(subject_list, subject_type):
    """
    Ensures that the subjects in the provided list are unique.

    Args:
        subject_list (list): The list of subjects to check.
        subject_type (str): The type of subjects being checked ('test' or 'validation').

    Raises:
        ValueError: If there are duplicate subjects in the list.
    """
    
    try:
        assert len(set(subject_list)) == len(subject_list), f"You have repeated {subject_type} subjects! Please verify your list of {subject_type} subjects."
        
    except AssertionError as error:
        print(colored(error, 'red'))
        sys.exit()