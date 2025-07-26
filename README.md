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

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face-recognition-classical.git
cd face-recognition-classical

### 2. Install Required Packages
bash
Copy
Edit
pip install -r requirements.txt

If you don't have a requirements.txt, the following packages are required:
bash
Copy
Edit
pip install numpy opencv-python scikit-image scikit-learn matplotlib

### 3. Prepare Dataset
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
<img width="940" height="394" alt="image" src="https://github.com/user-attachments/assets/a702e35a-6701-44d9-8767-6265a0dbca3c" />


🔬 Feature Extraction
Local Binary Patterns (LBP): captures micro-patterns of facial texture
<img width="880" height="631" alt="image" src="https://github.com/user-attachments/assets/9b7734a4-5ecb-4acc-b866-a6ce8345faca" />

Histogram of Oriented Gradients (HOG): encodes gradient direction changes
<img width="875" height="633" alt="image" src="https://github.com/user-attachments/assets/fdd5f418-a03a-4d2c-9e47-a487433b42f1" />

Gaussian Blur: reduces image noise for better feature stability
LBP + HOG Combined: concatenated into one unified vector
Saved as .npy files for efficient reuse in model training

📈 Model Evaluation
Classifier: KNeighborsClassifier(n_neighbors=3)
Validation: Stratified 5-Fold CV for balanced testing
Average Accuracy: 98.82%
<img width="772" height="863" alt="image" src="https://github.com/user-attachments/assets/4421020a-b6ba-4eaf-b8fe-a36f9f0c612f" />

Feature Vector Shape: 4366
Confusion Matrix: Provides per-class insight
<img width="940" height="853" alt="image" src="https://github.com/user-attachments/assets/f6b2d6c3-5520-4ee1-a356-49c08c020d04" />


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
<img width="940" height="710" alt="image" src="https://github.com/user-attachments/assets/6a6f1898-3551-47d8-a17b-2ecf978452a2" />
<img width="940" height="583" alt="image" src="https://github.com/user-attachments/assets/81d8bc06-75d4-4c51-9ed7-33f7f88b029b" />



🧑‍💻 Contributors
Ayyan Sohail — @ayyansohail
Sajjad Ahmed
Sammee Mudassar



