import torch.nn as nn
import torch.nn.functional as F

class AIImageDetectorCNN(nn.Module):
    def __init__(self):
        super(AIImageDetectorCNN, self).__init__()
        
        # Convolutional layers
        # Input: 3 x 128 x 128 (Assuming we resize images to 128x128)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 3 pooling layers: 128x128 -> 64x64 -> 32x32 -> 16x16
        # Flatten size: 128 channels * 16 * 16
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 1) # Output 1 for binary classification (Real vs Fake)

    def forward(self, x):
        # Layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Layer 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Layer 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC Layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
