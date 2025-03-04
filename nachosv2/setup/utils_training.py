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

