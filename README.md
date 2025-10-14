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
├── train_subset_mobilenet.py       # Main training script
├── realtime_fixed_roi_subset.py    # Real-time demo
├── evaluate_subset.py              # Model evaluation
├── model_evaluation.ipynb          # Comprehensive analysis
├── ML_PROJECT_REPORT.pdf          # Detailed project report
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam for real-time demo
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd hand-gesture-2
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
   ```

   Or install all at once:
   ```bash
   pip install tensorflow==2.20.0 opencv-python scikit-learn matplotlib seaborn numpy pandas jupyter
   ```

## 🎮 Usage

### 1. Real-time ASL Recognition Demo

**Start the live webcam demo:**
```bash
python realtime_fixed_roi_subset.py
```

**How to use:**
- Position your hand in the green ROI (Region of Interest) box
- Make ASL gestures for letters: A, B, C, F, K, Y
- See real-time predictions with confidence scores
- Press 'q' to quit

### 2. Model Evaluation

**Comprehensive analysis (Jupyter):**
```bash
jupyter notebook model_evaluation.ipynb
```

### 3. Train New Models

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

### Common Issues

**1. Import Errors**
```bash
# Install missing packages
pip install <package-name>
```

**2. Camera Not Working**
- Check camera permissions
- Ensure camera index is correct (usually 0)
- Try different camera if multiple available

**3. Model Loading Errors**
```bash
# Re-download or retrain models
python train_subset_mobilenet.py
```

**4. Low Performance**
- Ensure good lighting
- Position hand clearly in ROI
- Check camera quality

### Performance Optimization
- Use GPU if available: `pip install tensorflow-gpu`
- Reduce image size for faster inference
- Close other camera applications

## 📈 Results

### Real-time Performance
- **FPS**: 8-10 frames per second
- **Latency**: ~98ms per inference
- **Accuracy**: 95%+ in good lighting conditions

### Model Comparison
- **Standard**: 99.74% accuracy, high confidence
- **Robust**: 97%+ accuracy, better generalization
- **Label Smoothed**: ~89% accuracy, realistic confidence
