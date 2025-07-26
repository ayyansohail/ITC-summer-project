# Person Identification System

This project presents a **interpretable, face recognition pipeline** using image processing and machine learning. Instead of using deep learning, it relies on **Local Binary Patterns (LBP)** and **Histogram of Oriented Gradients (HOG)** for feature extraction and **K-Nearest Neighbors (KNN)** for classification. It also includes a **GUI with real-time webcam support** for easy testing.

> 📍 Final Project – Introduction to Computing  
> 📍 Institute of Business Administration, Karachi  
> 📍 Summer Semester 2025

---

##  Key Highlights

-  Face detection using Haar Cascades  
-  Gaussian blur noise reduction  
-  LBP + HOG feature vector fusion  
-  Stratified 5-Fold Cross Validation  
-  KNN classifier (k=3)  
-  Confusion matrix and performance metrics  
-  GUI with real-time prediction from webcam or uploaded image  

---

## 📁 Project Structure
├── eda.ipynb # Image validation and visualization
├── featureExtraction.ipynb # LBP, HOG, and LBP+HOG (with Gaussian blur)
├── model_training_and_evaluation.ipynb # KNN training, 5-fold validation, confusion matrix
├── face_recognition_gui.ipynb # GUI for real-time or static image testing
├── /data/
│ └── cropped_grayscaled_dataset/ # Folder-wise grayscale face images
├── /outputs/
│ ├── X_combined_denoised.npy
│ ├── y_combined_denoised.npy
│ └── label_map.npy
└── README.md


---

##  Details

1. Clone The repository
```bash
git clone https://github.com/yourusername/face-recognition-classical.git
cd face-recognition-classical

2. Install Required Packages
bash
Copy
Edit
pip install -r requirements.txt

If you don't have a requirements.txt, the following packages are required:
bash
Copy
Edit
pip install numpy opencv-python scikit-image scikit-learn matplotlib

3. Prepare Dataset
Organize face images like this:

bash
Copy
Edit
/data/cropped_grayscaled_dataset/
├── Person_A/
│   ├── image1.jpg
│   └── image2.jpg
├── Person_B/
│   └── image1.jpg


Images must be:
Grayscaled
Resized to 100x100
Faces cropped (using Haar cascades)

🔬 Feature Extraction
Local Binary Patterns (LBP): captures micro-patterns of facial texture

Histogram of Oriented Gradients (HOG): encodes gradient direction changes

Gaussian Blur: reduces image noise for better feature stability
LBP + HOG Combined: concatenated into one unified vector
Saved as .npy files for efficient reuse in model training

📈 Model Evaluation
Classifier: KNeighborsClassifier(n_neighbors=3)
Validation: Stratified 5-Fold CV for balanced testing
Average Accuracy: 98.82%

Feature Vector Shape: 4366
Confusion Matrix: Provides per-class insight


🖥️ GUI Interface
python
Copy
Edit
face_recognition_gui.ipynb
Upload an image OR
Use webcam live capture
Real-time prediction using trained KNN model
Handles preprocessing internally (grayscale, resize, denoise, feature extraction)

📸 GUI shows:
Predicted name
Optionally: confidence or distance
Auto-handles “Unknown” if below threshold

📊 Results
💯 Accuracy: 98.82% across 5 folds
🧪 Real-world test images achieved >82% accuracy with preprocessing
📉 Misclassifications on unseen data were addressed via:
Denoising
Confidence thresholding
Pipeline replication during inference



🧑‍💻 Contributors
Ayyan Sohail — @ayyansohail
Sajjad Ahmed
Sammee Mudassar



