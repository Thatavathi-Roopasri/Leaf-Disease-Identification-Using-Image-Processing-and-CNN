import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Config
MAIN_DATASET_DIR = "dataset"
ARCHIVE_DATASET_DIR = os.path.join("archive", "dataset")

MAIN_TRAIN_DIR = os.path.join(MAIN_DATASET_DIR, "train")
MAIN_VAL_DIR = os.path.join(MAIN_DATASET_DIR, "validation")
MAIN_TEST_DIR = os.path.join(MAIN_DATASET_DIR, "test")

ARCHIVE_TRAIN_DIR = os.path.join(ARCHIVE_DATASET_DIR, "train")
ARCHIVE_VAL_DIR = os.path.join(ARCHIVE_DATASET_DIR, "validation")
ARCHIVE_TEST_DIR = os.path.join(ARCHIVE_DATASET_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 6
PHASE3_EPOCHS = 8

PHASE1_LR = 3e-4
PHASE2_LR = 1e-5
PHASE3_LR = 5e-6

PHASE2_UNFREEZE_LAYERS = 20
PHASE3_UNFREEZE_LAYERS = 60

print("GPUs available:", tf.config.list_physical_devices("GPU"))


def archive_available(split_dir):
    return os.path.isdir(split_dir)


def make_generator(datagen, directory, class_labels=None, shuffle=True):
    return datagen.flow_from_directory(
        directory,
        classes=class_labels,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=shuffle,
    )


def alternating_generator(gen_a, gen_b):
    # Alternate batches from both datasets to expose both distributions each epoch.
    use_a = True
    while True:
        if use_a:
            yield next(gen_a)
        else:
            yield next(gen_b)
        use_a = not use_a


def setup_trainability(base_model, model, unfreeze_last_n):
    if unfreeze_last_n <= 0:
        base_model.trainable = False
    else:
        base_model.trainable = True
        freeze_until = max(len(base_model.layers) - unfreeze_last_n, 0)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in base_model.layers[freeze_until:]:
            layer.trainable = True

    # Keep classification head trainable in all phases.
    for layer in model.layers:
        if layer.name != base_model.name and not layer.name.startswith(("efficientnet", "input")):
            layer.trainable = True


def evaluate_on_generator(model, generator):
    generator.reset()
    y_pred = model.predict(generator, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = generator.classes
    return y_true_classes, y_pred_classes


print("\nInitializing enhanced data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)
test_val_datagen = ImageDataGenerator(rescale=1.0 / 255)

print("\nLoading main dataset generators...")
train_main = make_generator(train_datagen, MAIN_TRAIN_DIR, shuffle=True)
class_labels = list(train_main.class_indices.keys())
val_main = make_generator(test_val_datagen, MAIN_VAL_DIR, class_labels=class_labels, shuffle=False)
test_main = make_generator(test_val_datagen, MAIN_TEST_DIR, class_labels=class_labels, shuffle=False)

use_archive_train = archive_available(ARCHIVE_TRAIN_DIR)
use_archive_val = archive_available(ARCHIVE_VAL_DIR)
use_archive_test = archive_available(ARCHIVE_TEST_DIR)

train_generator = train_main
val_generator = val_main
test_generators = [test_main]

if use_archive_train:
    print("Loading archive train split and combining with main train split...")
    train_archive = make_generator(train_datagen, ARCHIVE_TRAIN_DIR, class_labels=class_labels, shuffle=True)
    train_generator = alternating_generator(train_main, train_archive)
else:
    train_archive = None

if use_archive_val:
    print("Loading archive validation split and combining with main validation split...")
    val_archive = make_generator(test_val_datagen, ARCHIVE_VAL_DIR, class_labels=class_labels, shuffle=False)
    val_generator = alternating_generator(val_main, val_archive)
else:
    val_archive = None

if use_archive_test:
    print("Loading archive test split for final evaluation...")
    test_archive = make_generator(test_val_datagen, ARCHIVE_TEST_DIR, class_labels=class_labels, shuffle=False)
    test_generators.append(test_archive)
else:
    test_archive = None

num_classes = len(class_labels)
print(f"Detected {num_classes} classes")

with open("label_mapping.json", "w") as f:
    json.dump(train_main.class_indices, f)

print("\nComputing class weights from combined training data...")
combined_train_classes = train_main.classes
if train_archive is not None:
    combined_train_classes = np.concatenate([combined_train_classes, train_archive.classes])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=combined_train_classes,
)
class_weight_dict = {i: float(w) for i, w in enumerate(weights)}

print("Class weights:")
for idx, cls_name in enumerate(class_labels):
    print(f"  {cls_name}: {class_weight_dict[idx]:.4f}")

train_samples_total = train_main.samples + (train_archive.samples if train_archive is not None else 0)
val_samples_total = val_main.samples + (val_archive.samples if val_archive is not None else 0)

steps_per_epoch = max(int(np.ceil(train_samples_total / BATCH_SIZE)), 1)
validation_steps = max(int(np.ceil(val_samples_total / BATCH_SIZE)), 1)

print(f"Train samples used: {train_samples_total}")
print(f"Validation samples used: {val_samples_total}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

print("\nBuilding EfficientNetB0 model...")
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
]

full_acc = []
full_val_acc = []
full_loss = []
full_val_loss = []

print("\n=== Phase 1 / 3: Train classifier head (base frozen) ===")
setup_trainability(base_model, model, unfreeze_last_n=0)
model.compile(
    optimizer=Adam(learning_rate=PHASE1_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
hist1 = model.fit(
    train_generator,
    epochs=PHASE1_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    class_weight=class_weight_dict,
    callbacks=callbacks,
)

print("\n=== Phase 2 / 3: Fine-tune last layers (light unfreeze) ===")
setup_trainability(base_model, model, unfreeze_last_n=PHASE2_UNFREEZE_LAYERS)
model.compile(
    optimizer=Adam(learning_rate=PHASE2_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
hist2 = model.fit(
    train_generator,
    epochs=PHASE2_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    class_weight=class_weight_dict,
    callbacks=callbacks,
)

print("\n=== Phase 3 / 3: Fine-tune deeper layers (broader unfreeze) ===")
setup_trainability(base_model, model, unfreeze_last_n=PHASE3_UNFREEZE_LAYERS)
model.compile(
    optimizer=Adam(learning_rate=PHASE3_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
hist3 = model.fit(
    train_generator,
    epochs=PHASE3_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    class_weight=class_weight_dict,
    callbacks=callbacks,
)

for hist in [hist1, hist2, hist3]:
    full_acc.extend(hist.history.get("accuracy", []))
    full_val_acc.extend(hist.history.get("val_accuracy", []))
    full_loss.extend(hist.history.get("loss", []))
    full_val_loss.extend(hist.history.get("val_loss", []))

print("\nSaving final model...")
model.save("cnn_model.keras")
print("Saved final model as cnn_model.keras")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(full_acc, label="Train Accuracy")
plt.plot(full_val_acc, label="Val Accuracy")
phase1_end = PHASE1_EPOCHS - 1
phase2_end = PHASE1_EPOCHS + PHASE2_EPOCHS - 1
plt.axvline(x=phase1_end, color="r", linestyle="--", label="Phase 1 -> 2")
plt.axvline(x=phase2_end, color="g", linestyle="--", label="Phase 2 -> 3")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(full_loss, label="Train Loss")
plt.plot(full_val_loss, label="Val Loss")
plt.axvline(x=phase1_end, color="r", linestyle="--", label="Phase 1 -> 2")
plt.axvline(x=phase2_end, color="g", linestyle="--", label="Phase 2 -> 3")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_graphs.png")
print("Saved training_graphs.png")
plt.close()

print("\nGenerating confusion matrix on combined test splits...")
all_y_true = []
all_y_pred = []

for test_gen in test_generators:
    y_true, y_pred = evaluate_on_generator(model, test_gen)
    all_y_true.append(y_true)
    all_y_pred.append(y_pred)

y_true_classes = np.concatenate(all_y_true)
y_pred_classes = np.concatenate(all_y_pred)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix (3-Phase Finetuned)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")
plt.close()

print("\nPer-class accuracy and classification report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))
print("\nProcess completed successfully!")
