from termcolor import colored
import sys


def check_unique_subjects(subject_list, subject_type):
    """
    Ensures that the subjects in the provided list are unique.

    Args:
        subject_list (list): The list of subjects to check.
        subject_type (str): The type of subjects being checked ('test' or 'validation').

    Raises:
        AssertionError: If there are duplicate subjects in the list.
    """

    # Checks if the subjects are unique
    assert len(set(subject_list)) == len(subject_list),\
    colored(f"You have repeated {subject_type} subjects! Please verify your list of {subject_type} subjects.",\
            'red')