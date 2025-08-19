import torch
import torch.nn as nn
import torch.nn.functional as F

class ShadowModel(nn.Module):
    """Convolutional Neural Network for CIFAR-10 classification."""
    def __init__(self):
        super(ShadowModel, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1)
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_layers = nn.ModuleList([
            nn.Linear(64 * 8 * 8, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 10)
        ])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.pool(F.relu(conv(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc_layers[0](x))
        x = self.dropout(x)
        x = F.relu(self.fc_layers[1](x))
        x = self.fc_layers[2](x)
        return x

class AttackModel(nn.Module):
    """MLP for membership inference attack"""
    def __init__(self, input_size=3):
        super(AttackModel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return self.sigmoid(x)