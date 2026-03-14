# Parking Occupancy Detection with Deep Learning

This project implements a computer vision system for detecting parking space occupancy using convolutional neural networks (CNNs). The system can analyze both images and videos to determine whether parking spaces are occupied or empty.

The project was developed as an experimental learning project to explore deep learning techniques in computer vision, particularly transfer learning and lightweight CNN architectures.

---

# Project Goals

The goals of this project are:

• Explore transfer learning using a pretrained CNN model  
• Implement a lightweight CNN trained from scratch  
• Compare model performance on a parking occupancy detection task  
• Build a simple inference pipeline for image and video detection  

---

# Models

Two models are implemented and compared:

##  VGG16 (Transfer Learning)

A pretrained VGG16 network is used as a feature extractor. The final layers are replaced with a small classifier for binary classification.

Advantages:

• strong pretrained visual features  
• faster convergence  

---

##  Lightweight CNN (From Scratch)

A small CNN architecture is implemented and trained from scratch as a baseline.

Advantages:

• significantly fewer parameters  
• faster training  
• competitive performance for this task  

---

# Machine Learning Pipeline

The system follows a typical machine learning workflow:

1. Data preprocessing (image resizing and normalization)
2. Model training using transfer learning or custom CNN
3. Model evaluation using accuracy, precision, recall and F1-score
4. Inference on images and videos
5. Model comparison on a shared test set

---

# Installation
Clone the repository:

```bash
git clone https://github.com/DraculaXS/parking-detection-project.git
cd parking-detection-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```
---

# Dataset Structure

Training data should be organized as:

    train_data/
    ├── train/
    │   ├── empty/      # images of empty parking spaces
    │   └── occupied/   # images of occupied parking spaces
    └── test/
        ├── empty/      # test images of empty spaces
        └── occupied/   # test images of occupied spaces

Each folder contains images of parking spaces.

---

# Training

Train the VGG16 model:

```bash
python train.py
```

Train the custom CNN:

```bash
python train_cnn.py
```

---

# Model Evaluation

Example evaluation metrics:

| Model | Accuracy |
|------|---------|
| VGG16 | 92.68% |
| CNN | 95.73% |

The lightweight CNN achieved slightly better performance in this dataset while requiring significantly fewer parameters.

---

# Model Comparison

The repository includes a comparison module that runs inference with multiple models on the same video stream and visualizes the results side-by-side.

```bash
python compare_models.py
```

---

# My Contributions

My main contributions include:

• Refactoring legacy Keras code to ensure compatibility with TensorFlow 2.x  
• Implementing a lightweight CNN baseline model  
• Developing model evaluation and comparison scripts  
• Structuring the project and improving documentation  

---

# Disclaimer

This project was developed for learning purposes and experimentation with deep learning techniques in computer vision.

---

# License

This project is released under the MIT License for the modifications and additional code contributed in this repository.

---

# Acknowledgements

This project was adapted from an earlier open-source parking detection implementation that served as a learning reference for this work.

The current repository updates and extends the original implementation using modern deep learning frameworks and additional model comparison experiments.
