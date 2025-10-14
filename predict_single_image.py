# predict_single_image.py - Single Image ASL Prediction
import cv2
import numpy as np
import tensorflow as tf
import json
import argparse
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_model_and_classes(model_path="models/asl_subset_mobilenet.h5", 
                          class_indices_path="models/class_indices.json"):
    """Load the trained model and class indices"""
    try:
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
        
        print(f"Loading class indices from: {class_indices_path}")
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        # Create inverse mapping for predictions
        inv_class_indices = {v: k for k, v in class_indices.items()}
        print(f"✅ Classes loaded: {list(class_indices.keys())}")
        
        return model, inv_class_indices
    
    except Exception as e:
        print(f"❌ Error loading model or classes: {e}")
        return None, None

def preprocess_image(image_path, target_size=(160, 160)):
    """Preprocess image for prediction"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, target_size)
        
        # Expand dimensions to create batch
        image = np.expand_dims(image, axis=0)
        
        # Apply MobileNetV2 preprocessing
        image = preprocess_input(image)
        
        return image
    
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def predict_image(model, image, class_mapping, top_k=3):
    """Make prediction on preprocessed image"""
    try:
        # Get prediction probabilities
        predictions = model.predict(image, verbose=0)
        
        # Get top k predictions
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        top_probabilities = predictions[0][top_indices]
        
        results = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probabilities)):
            letter = class_mapping.get(idx, f"Unknown_{idx}")
            confidence = prob * 100
            results.append({
                'rank': i + 1,
                'letter': letter,
                'confidence': confidence
            })
        
        return results
    
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None

def display_results(image_path, results):
    """Display prediction results"""
    print(f"\n🖼️  Image: {image_path}")
    print("=" * 50)
    
    if results:
        print("📊 Predictions:")
        for result in results:
            print(f"   {result['rank']}. Letter '{result['letter']}' - {result['confidence']:.2f}% confidence")
        
        # Highlight top prediction
        top_prediction = results[0]
        print(f"\n🎯 Top Prediction: {top_prediction['letter']} ({top_prediction['confidence']:.2f}%)")
        
        # Confidence assessment
        confidence = top_prediction['confidence']
        if confidence > 90:
            assessment = "🟢 Very Confident"
        elif confidence > 70:
            assessment = "🟡 Confident"
        elif confidence > 50:
            assessment = "🟠 Moderately Confident"
        else:
            assessment = "🔴 Low Confidence"
        
        print(f"📈 Confidence Level: {assessment}")
    else:
        print("❌ No predictions available")

def main():
    """Main function to handle command line arguments and run prediction"""
    parser = argparse.ArgumentParser(description='Predict ASL letter from single image')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', default='models/asl_subset_mobilenet.h5', 
                       help='Path to the model file')
    parser.add_argument('--classes', default='models/class_indices.json',
                       help='Path to class indices file')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file '{args.image_path}' not found")
        return
    
    # Load model and classes
    model, class_mapping = load_model_and_classes(args.model, args.classes)
    if model is None or class_mapping is None:
        return
    
    # Preprocess image
    print(f"\n🔄 Processing image: {args.image_path}")
    processed_image = preprocess_image(args.image_path)
    if processed_image is None:
        return
    
    # Make prediction
    print("🧠 Making prediction...")
    results = predict_image(model, processed_image, class_mapping, args.top_k)
    
    # Display results
    display_results(args.image_path, results)

# Example usage functions
def predict_from_path(image_path, model_path="models/asl_subset_mobilenet.h5", 
                     class_indices_path="models/class_indices.json"):
    """Simple function to predict from image path - for use in other scripts"""
    model, class_mapping = load_model_and_classes(model_path, class_indices_path)
    if model is None:
        return None
    
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None
    
    results = predict_image(model, processed_image, class_mapping, top_k=1)
    return results[0] if results else None

if __name__ == "__main__":
    print("🤟 ASL Single Image Prediction Tool")
    print("=" * 40)
    
    # Check if running with arguments
    import sys
    if len(sys.argv) == 1:
        print("📖 Usage Examples:")
        print("  python predict_single_image.py image.jpg")
        print("  python predict_single_image.py path/to/gesture.png --top-k 5")
        print("  python predict_single_image.py test_image.jpg --model models/asl_subset_mobilenet.h5")
        print("\n💡 Supported formats: .jpg, .jpeg, .png, .bmp")
        print("📂 Make sure your image shows a clear ASL hand gesture!")
    else:
        main()