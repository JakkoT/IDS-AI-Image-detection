import torch.nn as nn
import torch.nn.functional as F

class AIImageDetectorCNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for binary image classification (Real vs Fake).
    """
    def __init__(self):
        super(AIImageDetectorCNN, self).__init__()
        
        # Convolutional Layers (Feature Extraction)
        # We assume input images are resized to 128x128 pixels with 3 color channels (RGB).
        # Input Shape: [Batch_Size, 3, 128, 128]
        
        # Layer 1: 
        # Conv2d: Extracts low-level features (edges, colors).
        # out_channels=32: Creates 32 different feature maps.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # BatchNorm2d: Normalizes the output of the convolution. 
        # Helps training stability and speed by keeping activation distributions consistent.
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer 2:
        # Increases depth to 64 channels to capture more complex textures/patterns.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3:
        # Increases depth to 128 channels for high-level feature abstraction.
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling Layer: 
        # MaxPool2d reduces spatial dimensions (height/width) by half (stride=2).
        # This reduces computation and makes the model translation invariant (robust to position shifts).
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout: 
        # Randomly zeros out 50% of neurons during training.
        # Prevents overfitting by forcing the network to learn redundant representations.
        self.dropout = nn.Dropout(0.5)
        
        # Fully Connected Layers (Classification)
        # Calculate Flattened Input Size:
        # Original Image: 128x128
        # After Pool 1: 64x64
        # After Pool 2: 32x32
        # After Pool 3: 16x16
        # Final Tensor Shape before flattening: [Batch_Size, 128 (channels), 16, 16]
        # Flattened Vector Size = 128 * 16 * 16 = 32768
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        
        # Output Layer: 
        # Maps the 512 features to a single value (logit).
        # A positive value suggests one class (e.g., Real), negative suggests the other (Fake).
        self.fc2 = nn.Linear(512, 1) 

    def forward(self, x):
        """
        Defines the forward pass (data flow) of the network.
        Args:
            x: Input batch of images.
        Returns:
            x: Unnormalized output scores (logits).
        """
        
        # Block 1: Conv -> BN -> ReLU -> Pool
        # ReLU (Rectified Linear Unit) introduces non-linearity, allowing the model to learn complex functions.
        # without ReLU, the model would just be a linear regression.
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flattening:
        # Reshapes the 3D feature maps (Channels, Height, Width) into a 1D vector
        # so it can be fed into the Fully Connected (Dense) layers.
        # x.size(0) preserves the batch size. -1 infers the remaining dimension size.
        x = x.view(x.size(0), -1)
        
        # Classification Head:
        # FC1 -> ReLU -> Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Final Output (Logit)
        # We do NOT apply Sigmoid here because we use BCEWithLogitsLoss during training,
        # which applies Sigmoid internally for better numerical stability.
        x = self.fc2(x)
        
        return x