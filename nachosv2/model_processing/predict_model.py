import torch
from torch import nn
from torch.utils.data import DataLoader

# TODO: should it work with mixed precision?

def predict_model(execution_device: str, 
                  model: nn.Module,
                  dataloader: DataLoader):
    """
    Makes the model do predictions on the test dataset.

    Args:
        execution_device (str): The execution device.
        model (TrainingModel): The training model.
        test_dataloader (test_dataloader): The test_dataloader containing the test or validation data.

    Returns:
        predictions_list (list): The list of the label prediction for each image in the test data.
        true_labels_list (list): The list of the real label for each image in the test data.
        file_names_list (list): The list of the file names for each image in the test data.
    """
    
    # Initializations
    file_name_list = []
    prediction_list = []
    prediction_probabilities_list = []
    true_label_list = []

    with torch.set_grad_enabled(False):
        
        for inputs, labels, filepaths in dataloader:
            
            # Sends inputs and labels to the execution device  
            inputs = inputs.to(execution_device)
            labels = labels.to(execution_device)
            
            # Does the test
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            # Gets the predicted class indices
            _, class_predictions = torch.max(outputs, 1)

            # Formats the predictions and real labels into lists
            # file_names_list.append(file_names)
            prediction_list.extend(class_predictions.tolist())
            prediction_probabilities_list.extend(probabilities.tolist())
            true_label_list.extend(labels.tolist())
            file_name_list.extend(list(filepaths))

    return prediction_list, prediction_probabilities_list, \
           true_label_list, file_name_list
