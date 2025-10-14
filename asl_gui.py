# asl_gui.py - Simple ASL GUI with Upload and Live Demo
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import json
import threading
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class ASLRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤟 ASL Hand Gesture Recognition System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2563eb')
        
        # Initialize variables
        self.model = None
        self.class_mapping = None
        self.cap = None
        self.is_camera_running = False
        
        # Load model on startup
        self.load_model()
        
        # Create GUI elements
        self.create_header()
        self.create_tabs()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_model(self):
        """Load the ASL recognition model"""
        try:
            model_path = "models/asl_subset_mobilenet.h5"
            class_path = "models/class_indices.json"
            
            print("Loading ASL model...")
            self.model = tf.keras.models.load_model(model_path)
            
            with open(class_path, 'r') as f:
                class_indices = json.load(f)
            self.class_mapping = {v: k for k, v in class_indices.items()}
            
            self.model_status = "✅ Model loaded successfully"
            print(self.model_status)
            
        except Exception as e:
            self.model_status = f"❌ Error loading model: {str(e)}"
            print(self.model_status)
    
    def create_header(self):
        """Create the header section"""
        header_frame = tk.Frame(self.root, bg='#2563eb', height=80)
        header_frame.pack(fill='x', padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🤟 ASL Hand Gesture Recognition",
            font=('Arial', 24, 'bold'),
            fg='white',
            bg='#2563eb'
        )
        title_label.pack(pady=5)
        
        status_label = tk.Label(
            header_frame,
            text=self.model_status if hasattr(self, 'model_status') else "Loading model...",
            font=('Arial', 12),
            fg='#bfdbfe',
            bg='#2563eb'
        )
        status_label.pack()
    
    def create_tabs(self):
        """Create the tabbed interface"""
        # Create notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Tab 1: Upload Image
        self.create_upload_tab()
        
        # Tab 2: Live Demo
        self.create_demo_tab()
        
        # Tab 3: About
        self.create_about_tab()
    
    def create_upload_tab(self):
        """Create upload image tab"""
        upload_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(upload_frame, text="📁 Upload Image")
        
        # Instructions
        instruction_label = tk.Label(
            upload_frame,
            text="Click the button below to upload an ASL gesture image",
            font=('Arial', 16),
            fg='#374151',
            bg='white'
        )
        instruction_label.pack(pady=20)
        
        # Upload button
        upload_btn = tk.Button(
            upload_frame,
            text="📁 Select Image File",
            font=('Arial', 14, 'bold'),
            bg='#3b82f6',
            fg='white',
            padx=40,
            pady=20,
            command=self.upload_image
        )
        upload_btn.pack(pady=10)
        
        # Image display
        self.image_display = tk.Label(
            upload_frame,
            text="No image selected",
            font=('Arial', 14),
            fg='#6b7280',
            bg='#f9fafb',
            width=50,
            height=15
        )
        self.image_display.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Results area
        self.upload_results = tk.Text(
            upload_frame,
            height=8,
            font=('Arial', 12),
            bg='#f8fafc',
            fg='#374151'
        )
        self.upload_results.pack(pady=10, padx=20, fill='x')
    
    def create_demo_tab(self):
        """Create live demo tab with guaranteed button visibility"""
        demo_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(demo_frame, text="📹 Live Demo")
        
        # Title
        title_label = tk.Label(
            demo_frame,
            text="Live ASL Recognition Demo",
            font=('Arial', 18, 'bold'),
            fg='#1f2937',
            bg='white'
        )
        title_label.pack(pady=10)
        
        # Instructions
        instruction_label = tk.Label(
            demo_frame,
            text="Use the buttons below to start/stop the camera",
            font=('Arial', 14),
            fg='#6b7280',
            bg='white'
        )
        instruction_label.pack(pady=5)
        
        # Button frame - Make sure it's always visible
        button_frame = tk.Frame(demo_frame, bg='white', height=100)
        button_frame.pack(pady=20, fill='x')
        button_frame.pack_propagate(False)
        
        # Start button
        self.start_camera_btn = tk.Button(
            button_frame,
            text="📹 START CAMERA",
            font=('Arial', 16, 'bold'),
            bg='#10b981',
            fg='white',
            padx=50,
            pady=20,
            command=self.start_camera
        )
        self.start_camera_btn.pack(side='left', padx=20, pady=10)
        
        # Stop button
        self.stop_camera_btn = tk.Button(
            button_frame,
            text="⏹️ STOP CAMERA",
            font=('Arial', 16, 'bold'),
            bg='#ef4444',
            fg='white',
            padx=50,
            pady=20,
            command=self.stop_camera,
            state='disabled'
        )
        self.stop_camera_btn.pack(side='right', padx=20, pady=10)
        
        # Camera display
        self.camera_display = tk.Label(
            demo_frame,
            text="Camera view will appear here when started",
            font=('Arial', 14),
            fg='#6b7280',
            bg='#f3f4f6',
            width=80,
            height=20
        )
        self.camera_display.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Live prediction
        self.live_result = tk.Label(
            demo_frame,
            text="Predictions will appear here",
            font=('Arial', 18, 'bold'),
            fg='#1f2937',
            bg='white'
        )
        self.live_result.pack(pady=10)
        
        print("✅ Live Demo tab created with START/STOP buttons")
    
    def create_about_tab(self):
        """Create about tab"""
        about_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(about_frame, text="ℹ️ About")
        
        about_text = """
🤟 ASL Hand Gesture Recognition System

This application recognizes American Sign Language gestures for letters:
A, B, C, F, K, Y

Features:
• Upload images for instant recognition
• Real-time camera-based detection
• High accuracy predictions

How to use:
1. Upload Tab: Select an image file to analyze
2. Live Demo Tab: Use your webcam for real-time recognition
3. Make clear ASL gestures for best results

Technical Details:
• Built with TensorFlow and MobileNetV2
• 99.74% accuracy on test dataset
• Optimized for real-time performance

Developed by: C Panshul Reddy & C Yogesh Reddy
Institution: PES University
        """
        
        about_label = tk.Label(
            about_frame,
            text=about_text,
            font=('Arial', 12),
            fg='#374151',
            bg='white',
            justify='left'
        )
        about_label.pack(pady=20, padx=20)
    
    def upload_image(self):
        """Handle image upload"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select ASL Gesture Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path)
                display_image = image.copy()
                display_image.thumbnail((400, 300))
                photo = ImageTk.PhotoImage(display_image)
                
                self.image_display.configure(image=photo, text="")
                self.image_display.image = photo
                
                # Process prediction
                img_array = np.array(image.resize((160, 160)))
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                
                img_batch = np.expand_dims(img_array, axis=0)
                img_preprocessed = preprocess_input(img_batch)
                
                predictions = self.model.predict(img_preprocessed, verbose=0)
                
                # Show results
                self.upload_results.delete(1.0, tk.END)
                self.upload_results.insert(tk.END, f"📁 File: {os.path.basename(file_path)}\n\n")
                self.upload_results.insert(tk.END, "🎯 Top Predictions:\n")
                
                for i in range(min(3, len(predictions[0]))):
                    idx = np.argsort(predictions[0])[::-1][i]
                    letter = self.class_mapping.get(idx, f"Unknown_{idx}")
                    confidence = predictions[0][idx] * 100
                    
                    emoji = "🟢" if confidence > 80 else "🟡" if confidence > 60 else "🔴"
                    self.upload_results.insert(tk.END, f"{i+1}. {emoji} Letter '{letter}' - {confidence:.1f}%\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def start_camera(self):
        """Start camera"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot access camera!")
                return
            
            self.is_camera_running = True
            self.start_camera_btn.config(state='disabled')
            self.stop_camera_btn.config(state='normal')
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            print("✅ Camera started successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop camera"""
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
        
        self.start_camera_btn.config(state='normal')
        self.stop_camera_btn.config(state='disabled')
        
        self.camera_display.configure(
            image="",
            text="Camera stopped. Click START CAMERA to begin again."
        )
        self.live_result.config(text="Predictions will appear here")
        
        print("✅ Camera stopped")
    
    def camera_loop(self):
        """Camera processing loop"""
        while self.is_camera_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Draw ROI
                h, w = frame.shape[:2]
                roi_x, roi_y = w//2 - 100, h//2 - 100
                roi_w, roi_h = 200, 200
                
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
                cv2.putText(frame, "Hand here", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Extract ROI for prediction
                roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                
                if roi.size > 0:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_resized = cv2.resize(roi_rgb, (160, 160))
                    roi_batch = np.expand_dims(roi_resized, axis=0)
                    roi_preprocessed = preprocess_input(roi_batch)
                    
                    predictions = self.model.predict(roi_preprocessed, verbose=0)
                    top_idx = np.argmax(predictions[0])
                    confidence = predictions[0][top_idx] * 100
                    letter = self.class_mapping.get(top_idx, f"Unknown_{top_idx}")
                    
                    if confidence > 60:
                        emoji = "🟢" if confidence > 80 else "🟡"
                        result_text = f"{emoji} Letter: {letter} ({confidence:.1f}%)"
                    else:
                        result_text = "🔴 Show clearer gesture"
                    
                    # Update display (thread-safe)
                    self.root.after(0, lambda: self.live_result.config(text=result_text))
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil.thumbnail((640, 480))
                photo = ImageTk.PhotoImage(frame_pil)
                
                self.root.after(0, lambda p=photo: self.update_camera_display(p))
                
            except Exception as e:
                print(f"Camera error: {e}")
                break
    
    def update_camera_display(self, photo):
        """Update camera display"""
        if self.is_camera_running:
            self.camera_display.configure(image=photo, text="")
            self.camera_display.image = photo
    
    def on_closing(self):
        """Handle app closing"""
        if self.is_camera_running:
            self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ASLRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()