from statistics import mode
from termcolor import colored


def check_and_convert_labels(labels):
    """
    TODO
    """
    
    if type(labels) == dict:
        labels_list = list(labels.values())
        
    elif type(labels) == list:
        labels_list = labels
        
    else:
        raise ValueError(colored("Labels must be either a dictionary or a list", 'red'))

    return labels_list



def check_and_convert_values(values, labels_list):
    """
    TODO
    """
    values = pandadf_to_list(values)
    
    if not isinstance(values[0], str):
        
        # Initialization
        temp_values = []
        
        for val in values:
            
            try:
                label = labels_list[val]
                
            except KeyError:
                print(colored(f"Warning: The value {val} doesn't have a corresponding label correspondant.", 'yellow'))
                label = f"Unknown_{val}"
                
            temp_values.append(label)
            
        values = temp_values
    
    
    return values



def pandadf_to_list(dataframe):
    """
    TODO
    """
    
    # Initialization
    transformed_list = []
    
    for list in dataframe:
        transformed_list.append(list[0])
    
    return transformed_list



def find_shapes_mode(shapes, is_outer):
    """
    Finds the mode of all prediction shapes.
    
    Args:
        shapes (dict): Dictionary of shapes.
        is_outer (bool): Whether this is outer loop data.
    
    Returns:
        int: Mode of shapes.
    """
    
    # Initialization
    shapes_mode = []
    
    for configuration in shapes:
        
        for test_fold in shapes[configuration]:
            
            if not is_outer:
                
                for validation_fold in shapes[configuration][test_fold]:
                    shapes_mode.append(shapes[configuration][test_fold][validation_fold])
            
            else:
                shapes_mode.append(shapes[configuration][test_fold])
                
                
    return mode(shapes_mode)



def filter_matrices(matrices, shapes, shapes_mode, is_outer):
    """
    Filters matrices based on the mode shape.
    
    Args:
        matrices (dict): Dictionary of matrices.
        shapes (dict): Dictionary of shapes.
        shapes_mode (int): Mode of shapes.
        is_outer (bool): Whether this is outer loop data.
    
    Returns:
        dict: Filtered matrices.
    """
    
    for configuration in matrices:
        
        for test_fold in matrices[configuration]:
            
            # Initialization
            test_fold_matrices = []
            
            
            # Inner loop
            if not is_outer:
                
                for validation_fold in matrices[configuration][test_fold]:
                    validation_fold_shape = shapes[configuration][test_fold][validation_fold]
                    
                    if validation_fold_shape == shapes_mode:
                        test_fold_matrices.append(matrices[configuration][test_fold][validation_fold])
                        
                    else:
                        print(colored(f"Warning: Not including the validation fold {validation_fold} in the mean of ({configuration}, {test_fold})." +
                                      f"\n\tThe predictions expected to have {shapes_mode} rows, but got {validation_fold_shape} rows.\n", "yellow"))
                
                matrices[configuration][test_fold] = test_fold_matrices
            
            
            # Outer loop
            else:
                test_fold_shape = shapes[configuration][test_fold]
                
                if test_fold_shape == shapes_mode:
                    test_fold_matrices.append(matrices[configuration][test_fold])
                    matrices[configuration][test_fold] = test_fold_matrices
    
    
    return matrices


def get_matrices_of_mode_shape(shapes, matrices, is_outer):
    """
    Finds the mode length of all predictions that exist within a data folder.
    Checks if matrices should be considered in the mean value.

    Args:
        shapes (dict): The shapes (prediction rows) of the corresponding confusion matrices.
        matrices (dict): A dictionary of the matrices.
        is_outer (bool): Whether this data is of the outer loop.

    Returns:
        dict: A reduced dictionary of matrix-dataframes, organized by configuration and testing fold.
    """
    
    shapes_mode = find_shapes_mode(shapes, is_outer)
    
    filtered_matrices = filter_matrices(matrices, shapes, shapes_mode, is_outer)
    
    
    return filtered_matrices
