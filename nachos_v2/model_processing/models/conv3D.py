import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightweightBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class Conv3DModel(nn.Module):
    def __init__(self, configuration_file):
        super(Conv3DModel, self).__init__()
        
        self.model_type = configuration_file["selected_model_name"]
        
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        
        self.block1 = LightweightBlock(16, 16)
        self.block2 = LightweightBlock(16, 32)
        self.block3 = LightweightBlock(32, 32)
        
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc = nn.Linear(32, 3)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.block1(out)
        out = F.max_pool3d(out, 2)
        out = self.block2(out)
        out = F.max_pool3d(out, 2)
        out = self.block3(out)
        
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out