import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from model import AIImageDetectorCNN

# Configuration
DATA_DIR = 'archive'
BATCH_SIZE = 32
IMG_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
# specific device selection: Use GPU (cuda) if available for faster training, otherwise fallback to CPU.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data_loaders(data_dir, batch_size, img_size):
    """
    Prepares the training and validation data loaders.
    
    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Number of images per batch.
        img_size (int): Target size to resize images.
        
    Returns:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        class_to_idx: Dictionary mapping class names to indices.
    """
    # Data augmentation and normalization for training
    # - Resize: Ensures all images are the same size for the neural network input.
    # - ToTensor: Converts PIL images (0-255) to PyTorch tensors (0-1).
    # - Normalize: Standardizes pixel values to resemble ImageNet statistics (mean ~0, std ~1).
    #   This helps the model converge (learn) faster and reach a stable solution.
    #   Values (0.485, ...) are standard constants for pre-trained models/general natural images gathered from the internet.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset from folder structure (folder name = class label)
    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except FileNotFoundError:
        print(f"Error: Data directory '{data_dir}' not found.")
        return None, None, None

    # Split into train and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Dataset loaded. Classes: {full_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # DataLoader handles batching, shuffling, and parallel data loading (num_workers).
    # Shuffle=True for training prevents the model from learning order-based patterns.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, full_dataset.class_to_idx

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Executes the training loop.
    
    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function (BCEWithLogitsLoss).
        optimizer: Optimization algorithm (Adam).
        num_epochs: Number of times to iterate over the entire dataset.
        device: 'cuda' or 'cpu'.
    """
    best_acc = 0.0
    
    print(f"Starting training on {device}...")
    
    for epoch in range(num_epochs):
        # Set model to training mode. 
        # This enables layers like Dropout and BatchNorm that behave differently during training vs inference.
        model.train()
        
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # tqdm creates a visual progress bar for the epoch loop (wraps around the DataLoader)
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for inputs, labels in loop:
            # Move data to the configured device (GPU/CPU) for computation
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients.
            # PyTorch accumulates gradients by default. We must clear them before the backward pass
            # of this batch, otherwise gradients from previous batches would mix in.
            optimizer.zero_grad()
            
            # Forward Pass: Compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            
            # Prepare labels for Loss Calculation
            # BCEWithLogitsLoss expects labels to be Float tensors.
            # .unsqueeze(1) reshapes labels from [batch_size] to [batch_size, 1] to match the shape of the model output.
            labels = labels.float().unsqueeze(1) 
            
            # Calculate Loss: Determine how wrong the model's predictions are compared to actual labels
            loss = criterion(outputs, labels)
            
            # Backward Pass (Backpropagation):
            # Calculates the gradient of the loss with respect to model parameters.
            # It figures out "direction" and "magnitude" to adjust weights to reduce error.
            loss.backward()
            
            # Optimizer Step:
            # Updates the model's weights based on the computed gradients.
            optimizer.step()
            
            #Statistics & Metrics Tracking:
            
            # Add current batch loss to running total (loss.item() extracts the scalar value)
            running_loss += loss.item()
            
            # Convert logits (raw output) to probabilities using Sigmoid (0 to 1 range)
            probs = torch.sigmoid(outputs)
            
            # Threshold probabilities to binary predictions (0 or 1)
            preds = (probs > 0.5).float()
            
            # Count correct predictions
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            # Update progress bar with current batch loss
            loop.set_postfix(loss=loss.item())

        # Calculate average loss and accuracy for the epoch
        train_acc = correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        
        # Validation phase: Evaluate model on unseen data
        val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} -> "
              f"Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save the best model based on validation accuracy.
        # This ensures we keep the version of the model that generalized best, 
        # not necessarily the one from the very last epoch (which might be overfitting).
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model.")

    print("Training complete.")

def evaluate_model(model, loader, device):
    """
    Evaluates the model's performance on a given dataset (validation/test).
    """
    # Set model to evaluation mode.
    # Disables Dropout and switches BatchNorm to use running statistics.
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # torch.no_grad() disables gradient calculation.
    # We don't need gradients for evaluation, and this saves memory and computation.
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Move results back to CPU and convert to numpy for scikit-learn metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics using scikit-learn
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return acc, prec, rec, f1

def main():
    # Step 1: Prepare Data
    train_loader, val_loader, class_mapping = get_data_loaders(DATA_DIR, BATCH_SIZE, IMG_SIZE)
    
    if train_loader is None:
        return

    print(f"Class Mapping: {class_mapping}") # {'FAKE': 0, 'REAL': 1}

    # Step 2: Initialize Model
    model = AIImageDetectorCNN().to(DEVICE)
    
    # Step 3: Define Loss Function and Optimizer
    # BCEWithLogitsLoss combines Sigmoid layer + BCELoss in one class.
    # This is numerically more stable than using a plain Sigmoid followed by BCELoss.
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer: Adaptive Moment Estimation.
    # Generally performs better and converges faster than SGD for many deep learning tasks.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Step 4: Train
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)

    # Step 5: Final Evaluation
    print("\n--- Final Evaluation on Validation Set ---")
    # Load the best weights saved during training (not necessarily the last epoch's weights)
    model.load_state_dict(torch.load('best_model.pth'))
    acc, prec, rec, f1 = evaluate_model(model, val_loader, DEVICE)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()