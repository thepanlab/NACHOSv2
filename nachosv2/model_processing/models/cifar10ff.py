import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10FF(nn.Module):
    def __init__(self, configuration_file):
        super(CIFAR10FF, self).__init__()
        """
        Creates and prepares a model for training.
            
        Args:
            model_type (str): Name of the type of model to create.
            class_names (list of str): List of all classes. Use to know how many there are.
            
        Returns:
            model (nn.Module): The prepared torch.nn model.
        """

        # Sets the model definition
        self.model_type = configuration_file["selected_model_name"]

        input_channels = 3
        kernel_size = 5
        
        self.flatten = nn.Flatten()
        
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size)
        
        self._to_linear = None
        self._calculate_to_linear(torch.zeros(1, input_channels, 299, 299))

        
        self.fc1 = nn.Linear(self._to_linear, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(configuration_file["class_names"]))
    
    
    
    def _calculate_to_linear(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

    
    
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    