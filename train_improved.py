import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import defaultdict

# ============== CONFIG ==============
DATASET_DIR = "dataset"
ARCHIVE_DIR = "archive/dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

ARCHIVE_TRAIN_DIR = os.path.join(ARCHIVE_DIR, "train")
ARCHIVE_VAL_DIR = os.path.join(ARCHIVE_DIR, "validation")
ARCHIVE_TEST_DIR = os.path.join(ARCHIVE_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 3
FINETUNE_EPOCHS = 15
INITIAL_LR = 3e-4
FINETUNE_LR = 1e-5

print("GPUs available:", tf.config.list_physical_devices('GPU'))

# ============== ENHANCED DATA AUGMENTATION ==============
# More aggressive augmentation to handle similar diseases
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,           # Increased from 10
    zoom_range=0.2,              # Increased from 0.1
    horizontal_flip=True,
    vertical_flip=True,          # Added
    width_shift_range=0.15,      # Added
    height_shift_range=0.15,     # Added
    shear_range=0.1,             # Added
    brightness_range=[0.8, 1.2], # Added - helps with lighting variations
    fill_mode='nearest'
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

# ============== CREATE COMBINED DATASETS ==============
print("\n=== LOADING DATASETS ===")
print("Loading main dataset...")
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

# Create generators for archive dataset if available
train_combined = train_generator
val_combined = val_generator

archive_available = os.path.exists(ARCHIVE_TRAIN_DIR)
if archive_available:
    print("Loading archive dataset...")
    try:
        archive_train_generator = train_datagen.flow_from_directory(
            ARCHIVE_TRAIN_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
        
        archive_val_generator = test_val_datagen.flow_from_directory(
            ARCHIVE_VAL_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        # ============== COMBINE GENERATORS ==============
        def combine_generators(gen1, gen2):
            """Combine two generators by alternating batches"""
            while True:
                yield next(gen1)
                yield next(gen2)
        
        train_combined = combine_generators(train_generator, archive_train_generator)
        val_combined = combine_generators(val_generator, archive_val_generator)
        print("Archive dataset loaded and combined!")
    except Exception as e:
        print(f"Could not load archive dataset: {e}")
        print("Using main dataset only...")
else:
    print("Archive dataset not found, using main dataset only...")

num_classes = len(train_generator.class_indices)
class_labels = list(train_generator.class_indices.keys())

print(f"\nNumber of classes: {num_classes}")
print(f"Classes: {class_labels}")

# ============== CALCULATE CLASS WEIGHTS ==============
print("\n=== CALCULATING CLASS WEIGHTS ===")
# Calculate class weights to handle imbalance
class_weights_dict = {}

# Get all training file paths to calculate weights
class_counts = defaultdict(int)

for filename in train_generator.filenames:
    class_name = filename.split(os.sep)[0]
    class_idx = train_generator.class_indices[class_name]
    class_counts[class_idx] += 1

# Add archive counts if available
if archive_available:
    for filename in archive_train_generator.filenames:
        class_name = filename.split(os.sep)[0]
        class_idx = train_generator.class_indices[class_name]
        class_counts[class_idx] += 1

# Compute class weights using formula: num_samples / (num_classes * class_count)
total_samples = sum(class_counts.values())
for class_idx in range(num_classes):
    class_count = class_counts.get(class_idx, 1)
    class_weights_dict[class_idx] = total_samples / (num_classes * class_count)
    class_name = class_labels[class_idx]
    print(f"{class_name}: {class_weights_dict[class_idx]:.4f}")

# Calculate steps per epoch
steps_per_epoch_phase1 = max(len(train_generator.filenames) // BATCH_SIZE, 1)
steps_val = max(len(val_generator.filenames) // BATCH_SIZE, 1)

if archive_available:
    steps_per_epoch_phase1 = max(
        (len(train_generator.filenames) + len(archive_train_generator.filenames)) // (BATCH_SIZE * 2),
        1
    )
    steps_val = max(
        (len(val_generator.filenames) + len(archive_val_generator.filenames)) // (BATCH_SIZE * 2),
        1
    )

print(f"Steps per epoch (train): {steps_per_epoch_phase1}")
print(f"Steps per epoch (val): {steps_val}")

# Save label mapping
with open('label_mapping.json', 'w') as f:
    json.dump(train_generator.class_indices, f, indent=2)
print("\nSaved label mapping to label_mapping.json")

# ============== BUILD MODEL ==============
print("\n=== BUILDING MODEL ===")

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)  # Increased dropout
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)  # Added extra dropout
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("Model built successfully")
print(f"Total parameters: {model.count_params():,}")

# ============== PHASE 1: INITIAL TRAINING ==============
print("\n=== PHASE 1: INITIAL TRAINING (Base Model Frozen) ===")

model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for monitoring and early stopping
callbacks_phase1 = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'cnn_model_best_phase1.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Calculate steps per epoch for combined dataset
steps_per_epoch_phase1 = len(train_files) // BATCH_SIZE  # Use combined size

history_init = model.fit(
    train_combined,
    epochs=INITIAL_EPOCHS,
    validation_data=val_combined,
    steps_per_epoch=steps_per_epoch_phase1,
    validation_steps=steps_val,
    class_weight=class_weights_dict,
    callbacks=callbacks_phase1,
    verbose=1
)

model.save('cnn_model_initial_improved.keras')
print("Saved Phase 1 model as cnn_model_initial_improved.keras")

# ============== PHASE 2: FINE-TUNING ==============
print("\n=== PHASE 2: FINE-TUNING (Unfreeze Last 20 Layers) ===")

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Ensure top layers are trainable
for layer in model.layers:
    if layer.name != base_model.name and not layer.name.startswith(('efficientnet', 'input')):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    ),
    ModelCheckpoint(
        'cnn_model_best_phase2.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("Starting Phase 2 Fine-tuning...")
history_finetune = model.fit(
    train_combined,
    epochs=FINETUNE_EPOCHS,
    validation_data=val_combined,
    steps_per_epoch=steps_per_epoch_phase1,
    validation_steps=steps_val,
    class_weight=class_weights_dict,
    callbacks=callbacks_phase2,
    verbose=1
)

# ============== EVALUATION & VISUALIZATION ==============
print("\n=== EVALUATION ===")

model.save('cnn_model_improved.keras')
print("Saved final improved model as cnn_model_improved.keras")

# Combine histories
acc = history_init.history['accuracy'] + history_finetune.history['accuracy']
val_acc = history_init.history['val_accuracy'] + history_finetune.history['val_accuracy']
loss = history_init.history['loss'] + history_finetune.history['loss']
val_loss = history_init.history['val_loss'] + history_finetune.history['val_loss']

# Plot training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy', linewidth=2)
plt.plot(val_acc, label='Val Accuracy', linewidth=2)
plt.axvline(x=INITIAL_EPOCHS-1, color='r', linestyle='--', label='Start Fine-tuning')
plt.title('Training and Validation Accuracy (Improved Model)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss', linewidth=2)
plt.plot(val_loss, label='Val Loss', linewidth=2)
plt.axvline(x=INITIAL_EPOCHS-1, color='r', linestyle='--', label='Start Fine-tuning')
plt.title('Training and Validation Loss (Improved Model)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_graphs_improved.png', dpi=300)
print("Saved training_graphs_improved.png")
plt.close()

# ============== CONFUSION MATRIX & METRICS ==============
print("\nGenerating confusion matrix on test data...")

# Limit test predictions to avoid memory issues
test_samples = 0
test_generator.reset()
y_pred_all = []
y_true_all = []

for images, labels in test_generator:
    predictions = model.predict(images, verbose=0)
    y_pred_all.extend(np.argmax(predictions, axis=1))
    y_true_all.extend(np.argmax(labels, axis=1))
    test_samples += len(images)
    if test_samples >= 2000:  # Limit to 2000 samples
        break

y_pred_classes = np.array(y_pred_all)
y_true_classes = np.array(y_true_all)

# Calculate metrics
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels,
            cbar_kws={'label': 'Count'})
plt.title('Improved Confusion Matrix (with Class Weights & Enhanced Augmentation)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
print("Saved confusion_matrix_improved.png")
plt.close()

# Classification report
print("\n=== CLASSIFICATION REPORT ===")
report = classification_report(y_true_classes, y_pred_classes, target_names=class_labels)
print(report)

# Save report to file
with open('classification_report_improved.txt', 'w') as f:
    f.write("IMPROVED MODEL - CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

# ============== IDENTIFY CONFUSION PAIRS ==============
print("\n=== CONFUSION ANALYSIS ===")
print("Top 10 Confusion Pairs (Most Misclassifications):\n")

confusion_pairs = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and cm[i][j] > 0:
            confusion_pairs.append((cm[i][j], class_labels[i], class_labels[j]))

confusion_pairs.sort(reverse=True)
for count, true_label, pred_label in confusion_pairs[:10]:
    print(f"{count:3d}x — {true_label} misclassified as {pred_label}")

print("\n✓ Training complete! Check the generated files for detailed analysis.")
