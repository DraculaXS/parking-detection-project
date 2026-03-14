# compare_models.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set parameters
img_width, img_height = 48, 48
validation_data_dir = "train_data/test"
batch_size = 32

# Data generator
datagen = ImageDataGenerator(rescale=1./255)

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False  # Must be False to maintain order
)

# Load both models
print("Loading VGG16 model...")
vgg_model = load_model('VGG16_model.keras')

print("Loading Simple CNN model...")
cnn_model = load_model('cnn_model.keras')

# Predict
print("Predicting...")
vgg_pred = vgg_model.predict(validation_generator)
cnn_pred = cnn_model.predict(validation_generator)

vgg_pred_classes = np.argmax(vgg_pred, axis=1)
cnn_pred_classes = np.argmax(cnn_pred, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# Print classification reports
print("\n" + "="*60)
print("VGG16 Model Evaluation Report")
print("="*60)
print(classification_report(true_classes, vgg_pred_classes, target_names=class_labels))

print("\n" + "="*60)
print("Simple CNN Model Evaluation Report")
print("="*60)
print(classification_report(true_classes, cnn_pred_classes, target_names=class_labels))

# Calculate accuracy
vgg_acc = np.mean(vgg_pred_classes == true_classes)
cnn_acc = np.mean(cnn_pred_classes == true_classes)

print("\n" + "="*60)
print("Model Comparison Summary")
print("="*60)
print(f"VGG16 Accuracy: {vgg_acc:.4f}")
print(f"Simple CNN Accuracy: {cnn_acc:.4f}")
print(f"Accuracy Difference: {abs(vgg_acc - cnn_acc):.4f}")

# Plot confusion matrices for comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# VGG16 confusion matrix
cm_vgg = confusion_matrix(true_classes, vgg_pred_classes)
sns.heatmap(cm_vgg, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
axes[0].set_title(f'VGG16 Confusion Matrix (Accuracy: {vgg_acc:.3f})')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# CNN confusion matrix
cm_cnn = confusion_matrix(true_classes, cnn_pred_classes)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels, ax=axes[1])
axes[1].set_title(f'Simple CNN Confusion Matrix (Accuracy: {cnn_acc:.3f})')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

print("\n Result plot saved as model_comparison.png")