# Parking Occupancy Detection with Deep Learning

This project implements a computer vision system for detecting parking space occupancy using convolutional neural networks (CNNs). The system can analyze both images and videos to determine whether parking spaces are occupied or empty.

The project was developed as an experimental learning project to explore deep learning techniques in computer vision, particularly transfer learning and lightweight CNN architectures.

---

## Project Goals

The goals of this project are:

• Explore transfer learning using a pretrained CNN model  
• Implement a lightweight CNN trained from scratch  
• Compare model performance on a parking occupancy detection task  
• Build a simple inference pipeline for image and video detection  

---

## Models

Two models are implemented and compared:

###  VGG16 (Transfer Learning)

A pretrained VGG16 network is used as a feature extractor. The final layers are replaced with a small classifier for binary classification.

Advantages:

• strong pretrained visual features  
• faster convergence  

---

###  Lightweight CNN (From Scratch)

A small CNN architecture is implemented and trained from scratch as a baseline.

Advantages:

• significantly fewer parameters  
• faster training  
• competitive performance for this task  

---

## Machine Learning Pipeline

The system follows a typical machine learning workflow:

1. Data preprocessing (image resizing and normalization)
2. Model training using transfer learning or custom CNN
3. Model evaluation using accuracy, precision, recall and F1-score
4. Inference on images and videos
5. Model comparison on a shared test set

---

## Installation
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

## Dataset Structure

Training data should be organized as follows:

    train_data/
    ├── train/
    │   ├── empty/      # images of empty parking spaces
    │   └── occupied/   # images of occupied parking spaces
    └── test/
        ├── empty/      # test images of empty spaces
        └── occupied/   # test images of occupied spaces

Each folder contains images of parking spaces.

---

## Training

Train the VGG16 model:

```bash
python train.py
```

Train the custom CNN:

```bash
python train_cnn.py
```

---
## Inference and Model Comparison

Run detection and compare the performance of the two models.

Compare both models (image + video)
```bash
python park_test.py --mode compare --test_type both
```

Run detection using only VGG16
```bash
python park_test.py --mode vgg --test_type both
```

Run detection using only the custom CNN
```bash
python park_test.py --mode cnn --test_type both
```

Compare models on images only
```bash
python park_test.py --mode compare --test_type image
```

Compare models on videos only
```bash
python park_test.py --mode compare --test_type video
```

---

## Model Evaluation

model_comparison.py evaluates the trained models on the test dataset and compares their performance using accuracy, classification reports, and confusion matrices.

```bash
python compare_models.py
```

Example evaluation metrics:

| Model | Accuracy |
|------|---------|
| VGG16 | 92.68% |
| CNN | 95.73% |

The lightweight CNN achieved slightly better performance in this dataset while requiring significantly fewer parameters.

---

## My Contributions

My main contributions include:

• Refactoring legacy Keras code to ensure compatibility with TensorFlow 2.x  
• Implementing a lightweight CNN baseline model  
• Developing model evaluation and comparison scripts  
• Structuring the project and improving documentation  

---

## Disclaimer

This project was developed for learning purposes and experimentation with deep learning techniques in computer vision.

---

## License

This project is released under the MIT License for the modifications and additional code contributed in this repository.

---

## Acknowledgements

This project was adapted from an earlier open-source parking detection implementation that served as a learning reference for this work.

The current repository updates and extends the original implementation using modern deep learning frameworks and additional model comparison experiments.
