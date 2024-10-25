import torch


def predict_model(execution_device, model, test_dataloader):
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
    
    # Sets the model to evaluation mode
    model.eval()
    
    # Initializations
    file_names_list = []
    predictions_labels_list = []
    predictions_probabilities_list = []
    true_labels_list = []


    with torch.no_grad():
        
        for inputs, labels, _, file_names in test_dataloader:
            
            # Sends inputs and labels to the execution device  
            inputs = inputs.to(execution_device)
            labels = labels.to(execution_device)
            
            # Does the test
            probabilities_output_tensor = model(inputs)
            
            # Gets the predicted class indices
            _, predicted_indices = torch.max(probabilities_output_tensor, 1)

            # Formats the predictions and real labels into lists
            file_names_list.append(file_names)
            predictions_labels_list.extend(predicted_indices.tolist())
            predictions_probabilities_list.extend(probabilities_output_tensor.tolist())
            true_labels_list.extend(labels.tolist())


    return predictions_labels_list, predictions_probabilities_list, true_labels_list, file_names_list
