import torch
from torchvision import transforms
from PIL import Image
import sys
import os
from model import AIImageDetectorCNN

# Configuration (Must match training)
IMG_SIZE = 128
MODEL_PATH = 'best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    """Loads the trained model architecture and weights."""
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Have you trained the model yet?")
        sys.exit(1)

    model = AIImageDetectorCNN()
    
    # Load weights (map_location ensures it works on CPU if trained on GPU)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)
        
    model.to(DEVICE)
    model.eval() # Set to evaluation mode (important for BatchNorm/Dropout)
    return model

def preprocess_image(image_path):
    """Reads and transforms an image for the model."""
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)

    try:
        # Open and ensure RGB (handles PNGs with transparency or Grayscale)
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    # Same transforms as training
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Add batch dimension (C, H, W) -> (1, C, H, W)
    return transform(image).unsqueeze(0)

def predict(model, input_tensor):
    """Runs inference on the input tensor."""
    with torch.no_grad(): # Disable gradient calculation
        input_tensor = input_tensor.to(DEVICE)
        output = model(input_tensor)
        
        # Apply Sigmoid to get probability (0.0 to 1.0)
        prob = torch.sigmoid(output).item()
        
    return prob

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_image.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    print(f"Processing image: {image_path}")
    input_tensor = preprocess_image(image_path)
    
    print("Running prediction...")
    probability = predict(model, input_tensor)
    
    
    is_real = probability > 0.5
    label = "REAL" if is_real else "FAKE"
    confidence = probability if is_real else 1 - probability
    
    print("-" * 30)
    print(f"Result: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw Probability (Real): {probability:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
