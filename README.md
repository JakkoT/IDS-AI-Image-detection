# AI-Generated Image Detection using CNN

**Authors:** Jakko Turro, Markus Tõnson, Lauri Laud

---

## Project Overview

This project implements a **Convolutional Neural Network (CNN)** to detect AI-generated images and distinguish them from real photographs. The model is trained on a binary classification task (REAL vs FAKE) and achieves high accuracy in identifying synthetic images produced by AI image generators.

---

## Motivation & Goals

With the rapid advancement of AI image generation technologies (DALL-E, Midjourney, Stable Diffusion, etc.), it has become increasingly difficult to distinguish between real photographs and AI-generated images.

Our Goal is was to develop a deep learning model capable of automatically classifying images as either real or AI-generated with high accuracy and confidence.

---

## Repository Structure

```
IDS-AI-Image-detection/
│
├── archive/                    # Dataset directory
│   ├── FAKE/                   # AI-generated images
│   └── REAL/                   # Real photographs
│
├── model.ipynb                 #    Main Jupyter Notebook with full analysis
│                               #    Contains model definition, training,
│                               #    evaluation, and visualizations
│
├── validate_image.py           #    Image prediction script
│                               #    Classify a single image as REAL or FAKE
│
├── testing.py                  #    Utility script to check GPU/CUDA availability
│
├── best_model.pth              #    Saved model weights (best validation accuracy)
│
├── requirements.txt            #    Python dependencies
│
├── training_history.png        #    Training metrics visualization
├── confusion_matrix.png        #    Confusion matrix visualization
├── roc_curve.png               #    ROC curve visualization
├── final_metrics.png           #    Final performance metrics bar chart
│
├── C9_report.pdf               #    Project report document
│
└── README.md                   #    This file
```

### File Descriptions

| File | Description |
| `model.ipynb` | **Main notebook** - Contains the complete analysis pipeline including data loading, model training, evaluation with visualizations (confusion matrix, ROC curve, training history), and detailed explanations of each step. | |
| `validate_image.py` | Command-line tool to classify a single image. Outputs prediction (REAL/FAKE) with confidence score. |
| `testing.py` | Utility script to verify PyTorch installation and GPU availability. |
| `requirements.txt` | List of all Python packages required to run the project. |
| `archive/` | Dataset folder containing `FAKE/` and `REAL/` subdirectories with training images. |

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended but not required)
- 15GB disk space for the dataset

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/JakkoT/IDS-AI-Image-detection.git
   cd IDS-AI-Image-detection
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Verify GPU availability (optional):

   ```bash
   python testing.py
   ```

### Dataset Setup

The dataset should be organized in the following structure:

```
archive/
├── FAKE/    # Contains AI-generated images
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── REAL/    # Contains real photographs
    ├── photo1.jpg
    ├── photo2.png
    └── ...
```

If you have a zipped dataset (`archive.zip`), extract it to create this structure.

---

## Usage

### Training the Model

\*\*Using Jupyter Notebook

Open and run `model.ipynb` in Jupyter Notebook or VS Code. This provides:

- Step-by-step execution with explanations
- Interactive visualizations
- Ability to modify parameters easily

```bash
jupyter notebook model.ipynb
```

The trained model will be saved as `best_model.pth`.

### Running the Notebook

The notebook `model.ipynb` is organized into the following sections:

1. **Import Libraries** - Load all required packages
2. **Configuration** - Set hyperparameters (batch size, learning rate, epochs, etc.)
3. **Model Architecture** - Define the CNN structure
4. **Data Loading** - Load and preprocess images
5. **Training** - Train the model with progress tracking
6. **Visualization** - Plot training history, loss curves
7. **Evaluation** - Generate confusion matrix, ROC curve, and final metrics

Simply run all cells sequentially to replicate our analysis.

### Validating Images

To classify a single image:

```bash
python validate_image.py <path_to_image>
```

**Output:**

```
Loading model from best_model.pth...
Processing image: test_image.jpg
Running prediction...
------------------------------
Result: FAKE
Confidence: 94.32%
Raw Probability (Real): 0.0568
------------------------------
```

---

## Model Architecture

The CNN architecture consists of:

| Layer                    | Type                | Output Shape | Description                   |
| ------------------------ | ------------------- | ------------ | ----------------------------- |
| Input                    | -                   | 3×128×128    | RGB image                     |
| Conv1 + BN + ReLU + Pool | Convolutional Block | 32×64×64     | Low-level feature extraction  |
| Conv2 + BN + ReLU + Pool | Convolutional Block | 64×32×32     | Mid-level feature extraction  |
| Conv3 + BN + ReLU + Pool | Convolutional Block | 128×16×16    | High-level feature extraction |
| Flatten                  | -                   | 32768        | Reshape for FC layers         |
| FC1 + ReLU + Dropout     | Fully Connected     | 512          | Classification layer          |
| FC2                      | Output              | 1            | Logit output                  |

Key Components:

- BatchNormalization: Stabilizes training
- MaxPooling: Reduces spatial dimensions
- Dropout (50%): Prevents overfitting
- BCEWithLogitsLoss: Binary cross-entropy with numerical stability

---

## Results

After training, the model generates several visualization files:

| Visualization          | Description                                       |
| ---------------------- | ------------------------------------------------- |
| `training_history.png` | Loss and accuracy curves over epochs              |
| `confusion_matrix.png` | True/False positive/negative breakdown            |
| `roc_curve.png`        | ROC curve with AUC score                          |
| `final_metrics.png`    | Bar chart of Accuracy, Precision, Recall, F1, AUC |

The model achieves strong performance in distinguishing AI-generated images from real photographs. See the notebook for detailed metrics and analysis.
