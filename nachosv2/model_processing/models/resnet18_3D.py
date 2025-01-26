import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    """
    Block de base pour ResNet en 3D.
    Utilise deux convolutions 3D avec une connexion résiduelle.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample



    def forward(self, x):
        identity = x
        # Si un downsample est requis, l'appliquer
        if self.downsample is not None:
            identity = self.downsample(x)
        # Passage à travers les convolutions et la normalisation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Ajouter la connexion résiduelle
        out += identity
        out = self.relu(out)
        return out



class ResNet18_3D(nn.Module):
    def __init__(self, configuration_file):
        super(ResNet18_3D, self).__init__()
        """
        Implémentation de ResNet en 3D.
        Peut être configuré avec différents nombres de blocs de base.
        """
        
        # Sets the model definition
        self.model_type = configuration_file["architecture_name"]
        
        self.block = BasicBlock3D
        self.layers = [2, 2, 2, 2]
        
        self.in_channels = 64
        input_channels = 3
        
        # Convolution initiale
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
        
        # Création des couches de blocs
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride = 2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride = 2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride = 2)
        
        # Calculer la taille de l'image après les convolutions et le pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, len(configuration_file["class_names"]))



    def _make_layer(self, block, out_channels, blocks, stride = 1):
        downsample = None
        # Définir une couche de downsampling si nécessaire
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )
        layers = []
        # Ajouter le premier bloc avec downsampling
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        # Ajouter les blocs restants
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
