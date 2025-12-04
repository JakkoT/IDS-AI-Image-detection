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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data_loaders(data_dir, batch_size, img_size):
    # Data augmentation and normalization for training
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, full_dataset.class_to_idx

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_acc = 0.0
    
    print(f"Starting training on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            # Labels need to be float for BCEWithLogitsLoss (and same shape as output)
            labels = labels.float().unsqueeze(1) 
            
            loss = criterion(outputs, labels)
            
            # Backward + Optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            loop.set_postfix(loss=loss.item())

        train_acc = correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        
        # Validation
        val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} -> "
              f"Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model.")

    print("Training complete.")

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return acc, prec, rec, f1

def main():
    train_loader, val_loader, class_mapping = get_data_loaders(DATA_DIR, BATCH_SIZE, IMG_SIZE)
    
    if train_loader is None:
        return

    print(f"Class Mapping: {class_mapping}") # e.g., {'FAKE': 0, 'REAL': 1}

    model = AIImageDetectorCNN().to(DEVICE)
    
    # Binary Cross Entropy with Logits (more stable than BCELoss + Sigmoid)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)

    # Final detailed evaluation
    print("\n--- Final Evaluation on Validation Set ---")
    model.load_state_dict(torch.load('best_model.pth'))
    acc, prec, rec, f1 = evaluate_model(model, val_loader, DEVICE)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
