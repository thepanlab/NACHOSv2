import torch.nn as nn
import torch.nn.functional as F
import torch


class Simple3DCNN(nn.Module):
    def __init__(self, configuration_file):
        super(Simple3DCNN, self).__init__()
        """
        Initializes the Simple3DCNN model with layers based on the provided configuration file.

        Args:
            configuration_file (dict): A dictionary containing model configuration, including 'selected_model_name'.
        """
        
        # Sets the model definition
        self.model_type = configuration_file["selected_model_name"]
        
        self.debug = True
        
        # Defines the layers
        self.conv1 = nn.Conv3d(1, 16, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool = nn.MaxPool3d(kernel_size = 2, stride = 2, padding = 0)
        self.conv2 = nn.Conv3d(16, 32, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(32)
        self.fc1 = nn.Linear(32 * 46 * 52 * 46, 64)
        self.fc2 = nn.Linear(64, 3)



    def debug_print(self, x, layer_name):
        if self.debug:
            print(f"{layer_name} - Shape: {x.shape}, Min: {x.min().item():.4f}, Max: {x.max().item():.4f}, Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
            print(f"{layer_name} - NaN: {torch.isnan(x).any().item()}, Inf: {torch.isinf(x).any().item()}")



    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            x (torch.Tensor): Output tensor after passing through the network.
        """
        
        self.debug_print(x, "Input")
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        self.debug_print(x, "After Conv1 + BN + ReLU")

        x = self.pool(x)
        self.debug_print(x, "After Pool1")

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        self.debug_print(x, "After Conv2 + BN + ReLU")

        x = self.pool(x)
        self.debug_print(x, "After Pool2")

        x = x.view(x.size(0), -1)
        self.debug_print(x, "After Flatten")

        x = F.relu(self.fc1(x))
        self.debug_print(x, "After FC1")

        x = torch.clamp(x, min=0, max=100)
        self.debug_print(x, "After Clamp")

        x = self.fc2(x)
        self.debug_print(x, "Output")
        
        return x
