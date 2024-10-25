from termcolor import colored


def get_metrics_dictionary(list_of_metrics):
    """
    Gets the dictionary of metrics from the given list.
    
    Args:
        list_of_metrics (list of str): The list of wanted metrics.
    
    Return:
        metrics_dictionary (dict {key: bool}): The dictionary of metrics.
    """
    
    # Initialization of the dictionary
    metrics_dictionary = {
        'accuracy': True,
        'loss': True,
        'recall': False
    }
    
    
    # Verifies the list of metrics
    verified_list_of_metrics = verify_metrics_list(list_of_metrics, metrics_dictionary)
    
    
    # Changes the boolean of the metrics in the list
    for metric in verified_list_of_metrics:
        metrics_dictionary[metric] = True

    
    return metrics_dictionary



def verify_metrics_list(list_of_metrics, metrics_dictionary):
    """
    Verifies the list of metrics.
    
    Args:
        list_of_metrics (list of str): The list of wanted metrics.
        metrics_dictionary (dict {key: bool}): The dictionary of metrics.
    
    Return:
        verified_list_of_metrics (list of str): The list of metrics that will be used.
    """
    
    # Initializations
    authorized_list = metrics_dictionary.keys() # Correct metrics
    verified_list_of_metrics = []               # Returned list
    
    
    # Checks if the metrics are corrects
    for metric in list_of_metrics:
        
        if metric in authorized_list:
            verified_list_of_metrics.append(metric)
            
        else:
            print(colored(f"Warning: '{metric}' is not an authorized metric.", 'yellow'))
            print(colored(f"Authorized metrics: {list(authorized_list)}", 'yellow'))
    
    
    return verified_list_of_metrics