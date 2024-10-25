import pandas as pd
from termcolor import colored


def get_data(predictions_file_path, true_labels_file_path):
    """
    Reads in the labels and predictions from CSV.

    Args:
        predictions_file_path (str): The path to an indexed prediction file.
        true_labels_file_path (str): The patth to an indexed truth file.

    Returns:
        pandas.Dataframe: Two pandas dataframes of prediction and true values.
    """
    
    # Reads the CSV files
    try:
        predictions = pd.read_csv(predictions_file_path, header = None).to_numpy()
        
    except:
        print(colored(f"\nError: the predictions file was empty. \n\t{predictions_file_path}", 'red'))
        
        return None, None
    
    try:
        true_labels = pd.read_csv(true_labels_file_path, header = None).to_numpy()
        
    except:
        print(colored(f"\nError: the truth file was empty. \n\t{true_labels_file_path}", 'red'))
        
        return None, None

    
    # Gets the shapes
    predictions_rows = predictions.shape[0]
    true_labels_rows = true_labels.shape[0]

    
    # Makes the number of rows equal, in case uneven
    if predictions_rows > true_labels_rows:
        
        print(colored("Warning: The number of predicted values is greater than the true values in " +
                      f"{predictions_file_path.split('/')[-3]}: \n\tTrue: {true_labels_rows} | Predicted: {predictions_rows}",
                      'yellow'))
        
        predictions = predictions[:true_labels_rows, :]
                
        
    elif predictions_rows < true_labels_rows:
        
        print(colored("Warning: The number of true values is greater than the predicted values in " +
                      f"{predictions_file_path.split('/')[-3]}: \n\tTrue: {true_labels_rows} | Predicted: {predictions_rows}",
                      'yellow'))
        
        true_labels = true_labels[:predictions_rows, :]


    return true_labels, predictions