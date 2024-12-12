from typing import Dict, List, Tuple, Optional
from termcolor import colored

from nachosv2.data_processing import get_list_of_epochs


def generate_pairs(test_subject_list, validation_subject_list, do_shuffle, param_epoch, is_outer):
    """ Generates subject-subject pairs.
    
    Args:cvs
        test_subject_list (list of str): List of test subjects.
        validation_subject_list (list of str): List of validation subjects.
        validation_subject_list (list of str): List of all subjects.
        do_shuffle (bool): If the fold list should be shuffled or not.
    
    Return:
        (list of str tuples): A list of subject pairs.
    """
    
    l_epochs = get_list_of_epochs(param_epoch, is_outer, test_subject_list)
    print("generate_pairs")
    print("test_subject_list =", test_subject_list)
    print("l_epochs =", l_epochs)
    print("validation_subject_list =", validation_subject_list)
    
    # Generate subject-subject tuples
    folds = []
    for n_epochs, test_subject in zip(l_epochs, test_subject_list):
        
        # Outer loop: use test subjects only
        if validation_subject_list is None:   
            folds.extend([(n_epochs, test_subject, None)])
            
        # Inner loop: use validation subjects
        else:
            folds.extend(_get_training_combos(n_epochs, validation_subject_list, test_subject))
        
    # Shuffle the folds
    if do_shuffle:
        shuffle(folds)
        
    return folds   


def generate_list_folds_for_partitions(validation_fold_list: Optional[List[str]],
        fold_list: List[str],
        test_fold_name: str,
        do_shuffle: bool) -> List[Dict[str, List[str]]]:
    """ Generates folds for the subject.
    
    Args:
        fold_list (list of str): A list of fold names.
        validation_fold_list (list of str): A list of validation fold names.
        test_fold_name (str): The  test fold name.
        do_shuffle (bool): If the fold list should be shuffled or not.
        
    Returns:
        (list of dict): A list of folds, containing the subjects for testing, validation, and training.
        (int): The number of rotations for the training loop.
    """
    
    # If outer loop, compare the test subjects.
    if validation_fold_list is None:        
        partitions = [{
            'training': _fill_training_partition(fold_list, test_fold_name, None),
            'testing': [test_fold_name],
        }]
        
    # If inner loop, get the test-val combinations.
    else:        
        partitions = []
       
        for validation_fold_name in validation_fold_list:
            if validation_fold_name != test_fold_name:     
                partitions.append({
                    'training': _fill_training_partition(fold_list, test_fold_name, validation_fold_name), 
                    'validation': [validation_fold_name],
                    'testing': [test_fold_name],
                })
        
    # Shuffle the data and get the number of training rotations
    if do_shuffle:
        shuffle(partitions)

    return partitions


def generate_folds(test_subject_list, validation_subject_list, subject_list, test_subject, do_shuffle, validation_subject=None):
    """ Generates folds for the subject.
    
    Args:
        test_subject_list (list of str): A list of test subject names.
        validation_subject_list (list of str): A list of validation subject names.
        test_subject (str): The current test subject name.
        do_shuffle (bool): If the fold list should be shuffled or not.
        
    Returns:
        (list of dict): A list of folds, containing the subjects for testing, validation, and training.
        (int): The number of rotations for the training loop.
    """
    
    # If outer loop, compare the test subjects.
    if validation_subject_list is None:        
        folds = [{
            'testing': [test_subject],
            'training': _fill_training_fold(subject_list, test_subject, test_subject)
        }]
        
    # If inner loop, get the test-val combinations.
    else:        
        folds = []
        if validation_subject:    
            folds.append({
                'testing': [test_subject],
                'training': _fill_training_fold(validation_subject_list, test_subject, validation_subject), 
                'validation': [validation_subject]
            })
        
        else:
            for subject in validation_subject_list:
                if subject != test_subject:     
                    folds.append({
                        'testing': [test_subject],
                        'training': _fill_training_fold(validation_subject_list, test_subject, subject), 
                        'validation': [subject]
                    })
        
    # Shuffle the data and get the number of training rotations
    if do_shuffle:
        shuffle(folds)
    return folds, len(folds)


def _fill_training_partition(
        fold_list: List[str],
        test_fold_name: str,
        validation_fold_name: Optional[str]
        ) -> List[str]:
    """
    Fills the training fold for some subject.
    
    Args:
        subject_list (list of str): A list of possible subjects.
        test_subject (int): The testing subject.
        subject (int): The paired validation/testing subject.
        
    Returns:
        (list of str): A list of subjects in the training fold.
    """
    
    return [f for f in fold_list if f not in {test_fold_name, validation_fold_name}]


def _get_training_combos(n_epochs, subject_list, test_subject):
    """ Fills the training fold for some subject.
    Args:
        subject_list (list of str): A list of possible subjects.
        test_subject (int): The the testing subject.
        
    Returns:
        (list of str tuples): A list of subject pairs.
    """
    folds = []
    
    for subject in subject_list:
        
        if subject != test_subject:
            
            folds.append((n_epochs, test_subject, subject))
            
            
    return folds
