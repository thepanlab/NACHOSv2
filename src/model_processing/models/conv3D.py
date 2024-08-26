import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DModel(nn.Module):
    def __init__(self, configuration_file):
        super(Conv3DModel, self).__init__()
        """
        TODO
        """
        
        # Sets the model definition
        self.model_type = configuration_file["selected_model_name"]
        
        self.debug = False
        
        # Initializations
        width = configuration_file["image_size"][0]
        height = configuration_file["image_size"][1]
        depth = configuration_file["image_size"][2]
        

        self.conv1 = nn.Conv3d(1, 20, kernel_size=10, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(20)
        self.pool = nn.AvgPool3d(kernel_size=5, stride=5, padding=0)
        
        self.conv2 = nn.Conv3d(20, 20, kernel_size=10, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(20)
        
        self.conv3 = nn.Conv3d(20, 50, kernel_size=10, stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(50)
        
        
        # Calculates the size of the output after conv and pooling layers
        def conv3d_output_size(size, kernel_size, stride=1, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        def pool3d_output_size(size, kernel_size, stride=1, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        w = conv3d_output_size(width, 10)
        w = pool3d_output_size(w, 5, 5)
        w = conv3d_output_size(w, 10)
        w = conv3d_output_size(w, 10)

        h = conv3d_output_size(height, 10)
        h = pool3d_output_size(h, 5, 5)
        h = conv3d_output_size(h, 10)
        h = conv3d_output_size(h, 10)

        d = conv3d_output_size(depth, 10)
        d = pool3d_output_size(d, 5, 5)
        d = conv3d_output_size(d, 10)
        d = conv3d_output_size(d, 10)

        self.fc1 = nn.Linear(w * h * d * 50, 50)
        self.bn_fc1 = nn.BatchNorm1d(50)

        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(50, 10)
        self.bn_fc2 = nn.BatchNorm1d(10)
        
        self.fc3 = nn.Linear(10, 3)
        self.bn_fc3 = nn.BatchNorm1d(3)
    

    
    
    def debug_print(self, x, layer_name):
        if self.debug:
            print(f"{layer_name} - Shape: {x.shape}, Min: {x.min().item():.4f}, Max: {x.max().item():.4f}, Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
            print(f"{layer_name} - NaN: {torch.isnan(x).any().item()}, Inf: {torch.isinf(x).any().item()}")

    
    
    def forward(self, x):
        self.debug_print(x, "Input")
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn1(x)
        self.debug_print(x, "After Conv1 + ReLU + Pool + BN")
        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        self.debug_print(x, "After Conv2 + ReLU + BN")
        
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        self.debug_print(x, "After Conv3 + ReLU + BN")
        
        x = torch.flatten(x, 1)
        self.debug_print(x, "After Flatten")
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        self.debug_print(x, "After FC1 + BN + ReLU")
        
        x = self.dropout(x)
        self.debug_print(x, "Dropout")
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        self.debug_print(x, "After FC2 + BN + ReLU")
        
        x = F.relu(self.bn_fc3(self.fc3(x)))
        self.debug_print(x, "After FC3 + BN + ReLU")
        
        
        return x
