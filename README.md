# 🤟 ASL Hand Gesture Recognition System
**Real-Time American Sign Language Detection Using Deep Learning**

## 📋 Project Information

| **Field** | **Details** |
|-----------|-------------|
| **Project Title** | Real-Time ASL Hand Gesture Recognition System |
| **Course/Subject** | Machine Learning  |
| **Institution** | [PES University] |
| **Domain** | Computer Vision, Deep Learning, Accessibility Technology |

## 👥 Team Members

| **Name** | **Role** | **Responsibilities** | **Contact** |
|----------|----------|---------------------|-------------|
| **C Panshul Reddy** | Project Lead & ML Engineer | Model architecture, training |
| **C Yogesh Reddy** | Data Scientist | Dataset preparation, evaluation metrics  |


## 📖 Project Description

A comprehensive real-time American Sign Language (ASL) hand gesture recognition system that leverages state-of-the-art deep learning techniques to bridge communication gaps for the deaf and hard-of-hearing community. This project implements transfer learning with MobileNetV2 architecture to classify 6 fundamental ASL letters (A, B, C, F, K, Y) with exceptional accuracy and real-time performance.

### 🎯 **Objectives:**
- Develop an accurate ASL gesture recognition system (>95% accuracy)
- Implement real-time processing for practical applications
- Create an accessible tool for ASL learning and communication

### 🔬 **Research Focus:**
- **Transfer Learning**: Leveraging pre-trained MobileNetV2 for efficient training
- **Real-time Optimization**: Balancing accuracy and inference speed
- **Accessibility**: User-friendly interface for educational purposes

### 💡 **Innovation:**
- **Smart ROI Processing**: Fixed region of interest for consistent detection
- **Comprehensive Evaluation**: R² analysis, confidence calibration


## 🎯 Features

- **Real-time Recognition**: Live webcam-based ASL gesture detection
- **High Accuracy**: 99.74% accuracy on test dataset with advanced models
- **Transfer Learning**: Efficient MobileNetV2-based architecture
- **ROI Processing**: Fixed region of interest for consistent detection

## 🏆 Key Achievements

- ✅ **99.74% Test Accuracy** - State-of-the-art performance on ASL gesture classification
- ✅ **Real-time Processing** - 8-10 FPS with 98ms inference latency
- ✅ **Comprehensive Analysis** - Detailed evaluation with confidence metrics
- ✅ **Educational Impact** - Accessible tool for ASL learning community

## 🛠️ Technical Specifications

| **Component** | **Specification** |
|---------------|-------------------|
| **Framework** | TensorFlow 2.20.0, Keras |
| **Architecture** | MobileNetV2 + Custom Classifier |
| **Input Resolution** | 160×160×3 RGB images |
| **Model Size** | ~9.27 MB (optimized for mobile) |
| **Training Data** | 18,000 images (3,000 per class) |
| **Classes** | 6 ASL letters (A, B, C, F, K, Y) |
| **Inference Time** | 98ms per prediction |
| **Hardware Requirements** | CPU: 4GB RAM, GPU: Optional |

## 📊 Project Performance

| Model | Accuracy | R² Score | Use Case |
|-------|----------|----------|----------|
| Standard | 99.74% | 0.9974 | High accuracy scenarios |

## 📁 Project Structure

```
hand-gesture-2/
├── dataset_subset/          # Main dataset (6 ASL letters)
│   ├── train/              # Training images
│   ├── val/                # Validation images
│   └── test/               # Test images
├── dataset_balanced/        # Rebalanced dataset for better training
├── models/                 # Trained models
│   ├── asl_subset_mobilenet.h5      # Standard model
│   └── class_indices.json           # Class mappings
├── evaluation_results/      # Model evaluation outputs
├── asl_gui.py              # 🆕 Interactive GUI Application (Recommended)
├── train_subset_mobilenet.py       # Main training script
├── realtime_fixed_roi_subset.py    # Real-time webcam demo
├── predict_single_image.py         # 🆕 Single image prediction
├── batch_predict_images.py         # 🆕 Batch image processing
├── evaluate_subset.py              # Model evaluation
├── model_evaluation.ipynb          # Comprehensive analysis
├── gui_requirements.txt            # 🆕 GUI-specific dependencies
├── ML_PROJECT_REPORT.pdf          # Detailed project report
└── README.md               # This file
```

## 🚀 Quick Start

### 🎬 **Immediate Demo (GitHub Users)**

If you cloned from GitHub and have the model files:

```bash
# 1. Navigate to your actual project directory
# Common scenarios:

# If cloned directly:
cd hand-gesture-recognition  # or your cloned folder name

# If cloned into a test/parent folder:
cd test/ML_Project           # Your actual case
cd your-folder/project-name  # General case

# 2. Verify you're in the right location (should show models/, train_subset_mobilenet.py, etc.)
ls    # Linux/Mac
dir   # Windows

# 3. Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install tensorflow==2.20.0 opencv-python scikit-learn matplotlib seaborn numpy pandas

# 5. Run real-time demo (ensure you're in project root!)
python realtime_fixed_roi_subset.py
``` 

### 📥 **For New Users (Complete Setup)**

### Prerequisites

- Python 3.8+
- Webcam for real-time demo
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install tensorflow==2.20.0
   pip install opencv-python
   pip install scikit-learn
   pip install matplotlib
   pip install seaborn
   pip install numpy
   pip install pandas
   pip install jupyter
   pip install Pillow  # For GUI application
   ```

   Or install all at once:
   ```bash
   pip install tensorflow==2.20.0 opencv-python scikit-learn matplotlib seaborn numpy pandas jupyter Pillow
   ```

4. **Verify installation**
   ```bash
   # Check if model files exist
   ls models/  # Linux/Mac
   dir models\ # Windows
   
   # Should show: asl_subset_mobilenet.h5, class_indices.json
   ```

## 🎮 Usage

### 1. 🎨 Interactive GUI Application **NEW!** (Recommended)

**Launch the beautiful, user-friendly GUI:**
```bash
python asl_gui.py
```

**Features:**
- 📁 **Upload Image Tab**: Drag & drop or select images for instant prediction
- 📹 **Live Demo Tab**: Real-time webcam recognition with visual feedback
- ℹ️ **About Tab**: Complete project information and usage tips
- 🎯 **Smart Results**: Top predictions with confidence scores and color-coded feedback
- 🖥️ **Professional Interface**: Modern design with intuitive navigation

**Perfect for:**
- Educational demonstrations
- Testing multiple images quickly
- Presenting to audiences
- Learning ASL with immediate feedback

### 2. Real-time ASL Recognition Demo (Command Line)

**Start the live webcam demo:**
```bash
python realtime_fixed_roi_subset.py
```

**How to use:**
- Position your hand in the green ROI (Region of Interest) box
- Make ASL gestures for letters: A, B, C, F, K, Y
- See real-time predictions with confidence scores
- Press 'q' to quit

### 3. Single Image Prediction 🖼️

**Predict ASL letter from a single image:**
```bash
python predict_single_image.py your_image.jpg
```

**Advanced usage:**
```bash
# Show top 5 predictions
python predict_single_image.py gesture.png --top-k 5

# Use different model
python predict_single_image.py test.jpg --model models/asl_subset_mobilenet.h5
```

**Example output:**
```
🖼️  Image: test_gesture.jpg
==================================================
📊 Predictions:
   1. Letter 'A' - 94.32% confidence
   2. Letter 'F' - 3.21% confidence
   3. Letter 'C' - 1.89% confidence

🎯 Top Prediction: A (94.32%)
📈 Confidence Level: 🟢 Very Confident
```

### 4. Model Evaluation

**Comprehensive analysis (Jupyter):**
```bash
jupyter notebook model_evaluation.ipynb
```

### 5. Train New Models

**Train standard model:**
```bash
python train_subset_mobilenet.py
```

## 📋 Model Details

### Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 160x160x3
- **Classes**: 6 (A, B, C, F, K, Y)
- **Architecture**: Transfer Learning + Custom Classifier

### Training Configuration
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 12-25 (with early stopping)
- **Data Augmentation**: Rotation, shifts, zoom, flip

### Dataset
- **Total Images**: 18,000 (3,000 per class)
- **Split**: 70% train, 15% validation, 15% test
- **Balanced**: Equal samples per class
- **Preprocessing**: MobileNetV2 preprocessing

## 📊 Evaluation Metrics

The project includes comprehensive evaluation:
- **Accuracy**: Overall classification accuracy
- **R² Score**: Goodness of fit measure
- **Confusion Matrix**: Per-class performance
- **Classification Report**: Precision, recall, F1-score
- **Confidence Analysis**: Prediction uncertainty

## 🔧 Troubleshooting

### ⚠️ **IMPORTANT: Run from Correct Directory**

**Error:** `FileNotFoundError: Unable to open file 'models/asl_subset_mobilenet.h5'`

**Root Cause:** Running the script from the wrong directory.

**Common Directory Structures After Cloning:**

```bash
# Scenario 1: Direct clone
C:\Users\username\project-folder\
├── models/
├── realtime_fixed_roi_subset.py
└── ...

# Scenario 2: Cloned into parent folder (Your case!)
C:\Users\cpans\OneDrive\Desktop\test\ML_Project\
├── models/
├── realtime_fixed_roi_subset.py  
└── ...

# Scenario 3: Multiple nested folders
C:\Users\username\downloads\project\hand-gesture-recognition\
├── models/
├── realtime_fixed_roi_subset.py
└── ...
```

**Solutions for Each Scenario:**

```bash
# ❌ WRONG - Running from parent/wrong directory
C:\Users\cpans\OneDrive\Desktop\test> python realtime_fixed_roi_subset.py
C:\Users\cpans\OneDrive\Desktop> python ML_Project/realtime_fixed_roi_subset.py

# ✅ CORRECT - Navigate to actual project directory first
cd C:\Users\cpans\OneDrive\Desktop\test\ML_Project
python realtime_fixed_roi_subset.py
```

**Quick Directory Check:**
```bash
# You're in the RIGHT directory if you see these files:
dir   # Windows
ls    # Linux/Mac

# Should show:
# - models/ (folder)
# - realtime_fixed_roi_subset.py
# - train_subset_mobilenet.py  
# - dataset_subset/ (folder)
```

### Common Issues & Solutions

**1. Model File Not Found**
```bash
# Error: FileNotFoundError: models/asl_subset_mobilenet.h5
# Solution: Ensure you're in the correct project directory
cd hand-gesture-2  # or your project folder name
ls models/         # Check if model files exist
```

**2. Missing Models Folder Structure**
```bash
# Error: Cannot find path 'models\' because it does not exist
# Issue: Model files are in root directory instead of models/ folder

# Solution: Create proper folder structure
mkdir models
move asl_subset_mobilenet.h5 models\

# Create class_indices.json if missing
echo '{"A": 0, "B": 1, "C": 2, "F": 3, "K": 4, "Y": 5}' | Out-File -FilePath models\class_indices.json -Encoding UTF8

# Verify structure
dir models\   # Should show: asl_subset_mobilenet.h5, class_indices.json
```

**3. Missing Model Files Entirely**
If model files don't exist, download them from the GitHub repository or train new ones:
```bash
# Download from GitHub releases or train new model
python train_subset_mobilenet.py
```

**3. Virtual Environment Issues**
```bash
# Ensure virtual environment is activated
# Windows:
venv\Scripts\activate

# You should see (venv) in your terminal prompt
(venv) PS C:\Users\cpans\OneDrive\Desktop\hand-gesture-2>
```

**4. Import Errors**
```bash
# Install missing packages
pip install tensorflow==2.20.0
pip install opencv-python
pip install scikit-learn
```

**5. Camera Not Working**
- Check camera permissions in Windows settings
- Ensure no other applications are using the camera
- Try different camera index (0, 1, 2) in the code

**6. Model Loading Errors with Custom Loss Functions**
If you encounter issues with models trained with custom loss functions:
```bash
# Use the standard model instead
# Edit realtime_fixed_roi_subset.py and change:
MODEL = "models/asl_subset_mobilenet.h5"  # Use this line
```

### 📁 **Required Files Checklist**

Before running the real-time demo, ensure these files exist in the correct structure:

```
your-project-folder/
├── models/                           ← This folder must exist!
│   ├── asl_subset_mobilenet.h5      ← Main model file
│   └── class_indices.json           ← Class mappings
├── realtime_fixed_roi_subset.py     ← Demo script
├── train_subset_mobilenet.py        ← Training script
└── dataset_subset/                  ← Dataset folder (for training)
```

**⚠️ Common Issue: Model file in wrong location**

If your structure looks like this (WRONG):
```
your-project-folder/
├── asl_subset_mobilenet.h5          ← Model file in root (WRONG!)
├── realtime_fixed_roi_subset.py     
└── other files...
```

**Fix it with these commands:**
```bash
# Create models folder
mkdir models

# Move model file to correct location
move asl_subset_mobilenet.h5 models\

# Create class indices file
echo '{"A": 0, "B": 1, "C": 2, "F": 3, "K": 4, "Y": 5}' | Out-File -FilePath models\class_indices.json -Encoding UTF8

# Verify correct structure
dir models\
```



## 📈 Results

### Real-time Performance
- **FPS**: 8-10 frames per second
- **Latency**: ~98ms per inference
- **Accuracy**: 95%+ in good lighting conditions

### Model Comparison
- **Standard**: 99.74% accuracy, high confidence
- **Robust**: 97%+ accuracy, better generalization
- **Label Smoothed**: ~89% accuracy, realistic confidence
