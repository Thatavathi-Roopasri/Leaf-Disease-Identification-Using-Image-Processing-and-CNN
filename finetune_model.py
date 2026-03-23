import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Config
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
FINETUNE_EPOCHS = 10
LEARNING_RATE = 1e-5

print("GPUs available:", tf.config.list_physical_devices('GPU'))

# 1. Load the saved model
print("\nLoading model cnn_model.keras...")
model = load_model('cnn_model.keras')

# 2. Access the base EfficientNet model inside it
# 3. Unfreeze last 20 layers ONLY
# 4. Keep earlier layers frozen
print("Configuring layers for fine-tuning...")

base_model = None
for layer in model.layers:
    if isinstance(layer, Model) and 'efficientnet' in layer.name:
        base_model = layer
        break

if base_model:
    print(f"Found nested base model: {base_model.name}")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # Ensure classification head layers are also trainable
    for layer in model.layers:
        if layer.name != base_model.name:
            layer.trainable = True
else:
    print("Nested base model not found. Assuming flattened layers.")
    base_layers = []
    for layer in model.layers:
        if layer.name.startswith(('global_average_pooling', 'dense', 'dropout', 'flatten')):
            break
        base_layers.append(layer)
        
    print(f"Identified {len(base_layers)} base model layers.")
    for layer in base_layers[:-20]:
        layer.trainable = False
    for layer in base_layers[-20:]:
        layer.trainable = True
        
    start_top_layers = False
    for layer in model.layers:
        if layer.name.startswith(('global_average_pooling', 'dense', 'dropout', 'flatten')):
            start_top_layers = True
        if start_top_layers:
            layer.trainable = True

# 5. Compile model
print("\nCompiling model...")
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 7. Keep same dataset and generators
print("Initializing data generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)
test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = test_val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_labels = list(train_generator.class_indices.keys())

# 6. Train for 10 more epochs
print("\nStarting fine-tuning...")
history = model.fit(
    train_generator,
    epochs=FINETUNE_EPOCHS,
    validation_data=val_generator
)

# 8. Save updated model, graphs, confusion matrix
print("\nSaving updated model...")
model.save('cnn_model.keras')
print("Model saved as cnn_model.keras")

# Plot accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Fine-tuning Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Fine-tuning Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_graphs.png')
print("Saved training_graphs.png")
plt.close()

# Evaluate on test set
print("\nEvaluating fine-tuned model on test data...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Generate confusion matrix
print("\nGenerating new confusion matrix...")
test_generator.reset()
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = test_generator.classes

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (After Fine-tuning)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")
plt.close()

print("\nPer-class accuracy and classification report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))

print("\nFine-tuning completed successfully!")
