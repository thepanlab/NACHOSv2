import torch


def evaluate_model(execution_device, model, test_dataloader, loss_fn):
    """
    Evaluates the model on the test dataset.

    Args:
        execution_device (str): The execution device.
        model (TrainingModel): The training model.
        test_dataloader (test_dataloader): The test_dataloader containing the test data.
        loss_fn (nn.CrossEntropyLoss): The loss function.
        
    Returns:
        average_loss (float): The average loss of the model.
        accuracy (float): The accuracy of the model.
    """

    # Sets the model to evaluation mode
    model.eval()
    
    # Initializations
    correct_predictions = 0
    total_predictions = 0
    running_loss = 0.0
    
    
    with torch.no_grad():
        
        for inputs, labels, _, _ in test_dataloader:

            # Sends inputs and labels to the execution device
            inputs = inputs.to(execution_device)
            labels = labels.to(execution_device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Computes loss
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # Converts the outputs to predictions
            _, predictions = torch.max(outputs, 1)
            
            # Counts the number of correct predictions
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    
    
    # Calculates the average loss and accuracy
    average_loss = running_loss / len(test_dataloader.dataset)
    accuracy = correct_predictions / total_predictions
    
    print(f'test loss: {average_loss:.4f} | accuracy: {accuracy:.4f}')
    
    return average_loss, accuracy
