

def create_empty_history(metrics_dictionary):
    """
    TODO
    """
    
    # Creates the base history
    history = {'train_loss': [], 'validation_loss': [], 'train_accuracy': [], 'validation_accuracy': []}
    
    # Adds the wanted metrics to the history dictionary
    add_lists_to_history(history, metrics_dictionary)

    return history


def add_lists_to_history(history, metrics_dictionary):
    """
    Ajoute automatiquement des listes vides au dictionnaire `history` en fonction des `keys` données.
    TODO
    Args:
    history (dict): Le dictionnaire `history` auquel ajouter les listes.
    keys (list): Une liste de clés à ajouter au dictionnaire `history`.
    """
    
    # Gets the list of metrics
    keys_list = list(metrics_dictionary.keys())
    
    # Removes loss and accuracy from the list of metrics to add
    for key in ['loss', 'accuracy']:
        if key in keys_list:
            keys_list.remove(key)


    # Adds the keys in the history dictionary
    for key in keys_list:
        
        # If the key is asked ( = true in metrics_dictionary)
        if key in metrics_dictionary and metrics_dictionary[key]:
            
            if key not in history:
                history[key] = []
