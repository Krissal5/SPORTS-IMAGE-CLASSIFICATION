🏀 Sports Image Classification using Transfer Learning (EfficientNetB0)

📌 Project Overview

This project focuses on multi-class image classification using Transfer Learning.
A pre-trained EfficientNetB0 model is fine-tuned to classify sports images into their respective categories using deep learning techniques.

The goal is to build a moderately advanced Computer Vision project suitable for:

Learning Transfer Learning

Demonstrating real-world deep learning workflow

Showcasing on GitHub & resume

🧠 Problem Statement

Given an image of a sport, predict which sport category it belongs to using a deep learning model trained on labeled image data.

📂 Dataset Description

The dataset contains images organized into:

train/

valid/

test/

Each folder has subfolders per sport class

A CSV file (sports.csv) is provided for metadata reference

Data Structure:

sports_dataset/
│
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
├── valid/
├── test/
├── sports.csv

🛠️ Technologies Used

Python

TensorFlow / Keras

EfficientNetB0 (Transfer Learning)

Pandas & NumPy

Matplotlib

Google Colab (GPU)

🧩 Project Workflow
🔹 Step 1: Dataset Understanding

Inspected directory structure

Identified number of classes

Checked class distribution

Verified consistency across train/validation/test sets

🔹 Step 2: Environment Setup

Used Google Colab

Enabled GPU

Imported required libraries

🔹 Step 3: Data Loading

Loaded metadata using sports.csv

Extracted class labels

Verified dataset integrity

🔹 Step 4: Image Preprocessing

Resized images to 224×224

Applied normalization using EfficientNet preprocessing

Used data augmentation:

Rotation

Zoom

Horizontal flipping

🔹 Step 5: Model Building (Transfer Learning)

Loaded EfficientNetB0 pre-trained on ImageNet

Froze base layers

Added custom classification head:

Global Average Pooling

Dense layer

Dropout

Softmax output layer

🔹 Step 6: Model Training

Optimizer: Adam

Loss: Categorical Crossentropy

Metric: Accuracy

Trained using training data

Validated on validation set

🔹 Step 7: Model Evaluation

Evaluated performance on test set

Plotted:

Training vs Validation Accuracy

Training vs Validation Loss

Generated predictions

Analyzed confusion patterns

🔹 Step 8: Interpretation & Insights

Model generalized well due to transfer learning

Data augmentation helped reduce overfitting

Misclassifications occurred between visually similar sports

🔹 Step 9: Conclusion

EfficientNetB0 achieved strong performance with minimal training

Transfer learning significantly reduced training time

Project demonstrates end-to-end deep learning workflow

📊 Results

Achieved high validation accuracy

Stable learning curves

Effective generalization on unseen data

(Exact accuracy may vary based on training configuration)

🚀 Key Learnings

Transfer Learning using EfficientNet

Image preprocessing & augmentation

Multi-class classification

Deep learning project structuring

Model evaluation & interpretation

📁 Repository Structure
├── train/
├── valid/
├── test/
├── sports.csv
├── notebook.ipynb
├── EfficientNetB0_model.h5
└── README.md

💡 Future Improvements

Fine-tune upper EfficientNet layers

Try other architectures (ResNet, MobileNet)

Add Grad-CAM visualization

Deploy model as a web app

👤 Author

Krissal 

Aspiring Data Scientist | Machine Learning & Deep Learning Enthusiast

⭐ Acknowledgements

TensorFlow & Keras

ImageNet

Google Colab
