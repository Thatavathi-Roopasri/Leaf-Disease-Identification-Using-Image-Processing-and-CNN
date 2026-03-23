import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
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
EPOCHS = 10
LEARNING_RATE = 3e-4

print("GPUs available:", tf.config.list_physical_devices('GPU'))

# 2. Preprocessing & 3. Generators
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

num_classes = len(train_generator.class_indices)

# 5. Debug checks - Part 1
print("\n--- Debug Checks ---")
print("1. Class indices mapping:", train_generator.class_indices)

# 7. Output: Save label mapping
with open('label_mapping.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
print("Saved label mapping to label_mapping.json")

# Sample batch labels
x_batch, y_batch = next(train_generator)
print("2. Sample batch X shape:", x_batch.shape)
print("   Sample batch Y shape:", y_batch.shape)
print("   Sample batch Y labels (first 5):\n", y_batch[:5])

# 1. Model Structure
print("\nBuilding EfficientNetB0 model...")
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

# Freeze base model initially
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Training (Stage 1)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

print("\nStarting training (Stage 1)...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# 5. Debug checks - Part 2
print("\n3. Prediction probabilities for a batch:")
pred_probs = model.predict(x_batch[:5])
print("   Probabilities (first 5 samples):\n", pred_probs)

# 6. Evaluation
print("\nEvaluating model on test data...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Plot accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_graphs.png')
print("Saved training_graphs.png")
plt.close()

# Generate confusion matrix and classification report
print("\nGenerating confusion matrix...")
test_generator.reset()
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = test_generator.classes

class_labels = list(train_generator.class_indices.keys())

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")
plt.close()

print("\nPer-class accuracy and classification report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))

# 7. Output: Save model
model.save('cnn_model.keras')
print("Saved model as cnn_model.keras")

print("\nAll required steps completed successfully!")
