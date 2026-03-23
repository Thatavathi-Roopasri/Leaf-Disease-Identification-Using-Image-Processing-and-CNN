import os
import json
import math
from collections import Counter

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ===================== CONFIG =====================
MAIN_DATASET_DIR = "dataset"
ARCHIVE_DATASET_DIR = os.path.join("archive", "dataset")

TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"

IMAGE_SIZE = (320, 320)
BATCH_SIZE = 24
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 12
PHASE3_EPOCHS = 10

PHASE1_LR = 1e-3
PHASE2_LR = 3e-5
PHASE3_LR = 1e-5

GAMMA_FOCAL = 2.0
LABEL_SMOOTHING = 0.03

OUTPUT_MODEL_PATH = "cnn_model_targeted.keras"
OUTPUT_HISTORY_PLOT = "training_graphs_targeted.png"
OUTPUT_CM_PLOT = "confusion_matrix_targeted.png"
OUTPUT_REPORT_PATH = "classification_report_targeted.txt"
OUTPUT_LABEL_MAP = "label_mapping_targeted.json"


# ===================== DATA DISCOVERY =====================
def split_dirs(split):
    dirs = [os.path.join(MAIN_DATASET_DIR, split)]
    archive_dir = os.path.join(ARCHIVE_DATASET_DIR, split)
    if os.path.isdir(archive_dir):
        dirs.append(archive_dir)
    return dirs


def list_classes_from_main_train():
    main_train = os.path.join(MAIN_DATASET_DIR, TRAIN_SPLIT)
    if not os.path.isdir(main_train):
        raise FileNotFoundError(f"Missing training directory: {main_train}")
    classes = [d for d in os.listdir(main_train) if os.path.isdir(os.path.join(main_train, d))]
    classes = sorted(classes)
    if not classes:
        raise RuntimeError("No class directories found in main training split")
    return classes


def collect_split_samples(split, class_to_idx):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = []
    labels = []

    for root_dir in split_dirs(split):
        if not os.path.isdir(root_dir):
            continue

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            if class_name not in class_to_idx:
                continue

            class_id = class_to_idx[class_name]
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(exts):
                    paths.append(os.path.join(class_dir, fname))
                    labels.append(class_id)

    if not paths:
        raise RuntimeError(f"No images found for split: {split}")

    return paths, labels


def species_from_class_name(class_name):
    if class_name.startswith("Potato"):
        return "Potato"
    if class_name.startswith("Pepper"):
        return "Pepper"
    return "Tomato"


# ===================== TARGETED AUGMENTATION =====================
def random_resized_crop(image, min_scale=0.65, max_scale=1.0):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    scale = tf.random.uniform([], min_scale, max_scale)
    crop_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)

    crop_h = tf.maximum(crop_h, 32)
    crop_w = tf.maximum(crop_w, 32)

    image = tf.image.random_crop(image, size=[crop_h, crop_w, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


def sharpen(image):
    kernel = tf.constant([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    kernel = tf.repeat(kernel, repeats=3, axis=2)
    out = tf.nn.depthwise_conv2d(image[None, ...], kernel, strides=[1, 1, 1, 1], padding="SAME")
    return tf.clip_by_value(out[0], 0.0, 1.0)


def cutout(image, size=16):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    cy = tf.random.uniform([], minval=0, maxval=h, dtype=tf.int32)
    cx = tf.random.uniform([], minval=0, maxval=w, dtype=tf.int32)

    y1 = tf.maximum(0, cy - size // 2)
    y2 = tf.minimum(h, cy + size // 2)
    x1 = tf.maximum(0, cx - size // 2)
    x2 = tf.minimum(w, cx + size // 2)

    pad_top = y1
    pad_bottom = h - y2
    pad_left = x1
    pad_right = w - x2

    mask = tf.pad(
        tf.zeros([y2 - y1, x2 - x1, 3], dtype=image.dtype),
        [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        constant_values=1.0,
    )
    return image * mask


def preprocess_base(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def build_class_group_ids(class_to_idx):
    cross_plant = [
        "Potato___Late_blight",
        "Tomato_Late_blight",
    ]

    spot_cluster = [
        "Tomato__Target_Spot",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Early_blight",
        "Tomato_Bacterial_spot",
    ]

    texture_cluster = [
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato_Leaf_Mold",
        "Tomato__Tomato_mosaic_virus",
    ]

    def to_ids(names):
        out = [class_to_idx[n] for n in names if n in class_to_idx]
        return tf.constant(out, dtype=tf.int32) if out else tf.constant([], dtype=tf.int32)

    return to_ids(cross_plant), to_ids(spot_cluster), to_ids(texture_cluster)


def make_train_preprocess_fn(cross_ids, spot_ids, texture_ids):
    @tf.function
    def preprocess(image, disease_id, species_id):
        image = preprocess_base(image)

        in_cross = tf.reduce_any(tf.equal(disease_id, cross_ids))
        in_spot = tf.reduce_any(tf.equal(disease_id, spot_ids))
        in_texture = tf.reduce_any(tf.equal(disease_id, texture_ids))

        def cross_aug(img):
            img = random_resized_crop(img, min_scale=0.65, max_scale=1.0)
            img = tf.image.random_hue(img, max_delta=0.02)
            img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
            return tf.clip_by_value(img, 0.0, 1.0)

        def spot_aug(img):
            img = random_resized_crop(img, min_scale=0.75, max_scale=1.0)
            img = tf.image.random_contrast(img, lower=0.95, upper=1.2)
            img = tf.cond(tf.random.uniform([]) < 0.3, lambda: sharpen(img), lambda: img)
            img = tf.cond(tf.random.uniform([]) < 0.4, lambda: cutout(img, size=16), lambda: img)
            return tf.clip_by_value(img, 0.0, 1.0)

        def texture_aug(img):
            noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.01)
            img = tf.clip_by_value(img + noise, 0.0, 1.0)
            return img

        image = tf.cond(in_cross, lambda: cross_aug(image), lambda: image)
        image = tf.cond(in_spot, lambda: spot_aug(image), lambda: image)
        image = tf.cond(in_texture, lambda: texture_aug(image), lambda: image)

        image = tf.image.per_image_standardization(image)

        disease_onehot = tf.one_hot(disease_id, depth=num_classes)
        species_onehot = tf.one_hot(species_id, depth=num_species)

        return image, {"disease_output": disease_onehot, "species_output": species_onehot}

    return preprocess


@tf.function
def preprocess_eval(image, disease_id, species_id):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.image.per_image_standardization(image)

    disease_onehot = tf.one_hot(disease_id, depth=num_classes)
    species_onehot = tf.one_hot(species_id, depth=num_species)

    return image, {"disease_output": disease_onehot, "species_output": species_onehot}


# ===================== LOSS =====================
def effective_number_alpha(class_counts, beta=0.999):
    counts = np.array([class_counts[i] for i in range(len(class_counts))], dtype=np.float32)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / np.sum(weights) * len(counts)
    return weights.astype(np.float32)


def make_class_balanced_focal_loss(alpha, gamma=2.0, label_smoothing=0.03):
    alpha = tf.constant(alpha, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        n_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = y_true * (1.0 - label_smoothing) + label_smoothing / n_classes

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)

        alpha_factor = y_true * alpha
        focal_factor = tf.pow(1.0 - y_pred, gamma)

        loss_tensor = alpha_factor * focal_factor * ce
        return tf.reduce_sum(loss_tensor, axis=-1)

    return loss


# ===================== MODEL =====================
def build_model(num_classes, num_species):
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    disease_output = layers.Dense(num_classes, activation="softmax", name="disease_output")(x)
    species_output = layers.Dense(num_species, activation="softmax", name="species_output")(x)

    model = Model(inputs=base.input, outputs=[disease_output, species_output])
    return model, base


def set_unfreeze_ratio(base_model, ratio):
    total = len(base_model.layers)
    if ratio <= 0:
        for layer in base_model.layers:
            layer.trainable = False
        return

    trainable_start = int(total * (1.0 - ratio))
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= trainable_start


def compile_model(model, disease_loss, lr):
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss={
            "disease_output": disease_loss,
            "species_output": "categorical_crossentropy",
        },
        loss_weights={
            "disease_output": 1.0,
            "species_output": 0.3,
        },
        metrics={
            "disease_output": ["accuracy"],
            "species_output": ["accuracy"],
        },
    )


# ===================== PIPELINE =====================
print("GPUs available:", tf.config.list_physical_devices("GPU"))

tf.keras.utils.set_random_seed(SEED)

class_names = list_classes_from_main_train()
class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {i: name for name, i in class_to_idx.items()}

species_names = ["Pepper", "Potato", "Tomato"]
species_to_idx = {n: i for i, n in enumerate(species_names)}

num_classes = len(class_names)
num_species = len(species_names)

print(f"Classes ({num_classes}): {class_names}")
print(f"Species ({num_species}): {species_names}")

with open(OUTPUT_LABEL_MAP, "w", encoding="utf-8") as f:
    json.dump(class_to_idx, f, indent=2)

train_paths, train_labels = collect_split_samples(TRAIN_SPLIT, class_to_idx)
val_paths, val_labels = collect_split_samples(VAL_SPLIT, class_to_idx)
test_paths, test_labels = collect_split_samples(TEST_SPLIT, class_to_idx)

train_species = [species_to_idx[species_from_class_name(class_names[i])] for i in train_labels]
val_species = [species_to_idx[species_from_class_name(class_names[i])] for i in val_labels]
test_species = [species_to_idx[species_from_class_name(class_names[i])] for i in test_labels]

class_counts_raw = Counter(train_labels)
class_counts = {i: class_counts_raw.get(i, 1) for i in range(num_classes)}
alpha = effective_number_alpha(class_counts, beta=0.999)

print("Class counts:")
for i in range(num_classes):
    print(f"  {class_names[i]}: {class_counts[i]}")

cross_ids, spot_ids, texture_ids = build_class_group_ids(class_to_idx)


def decode_image(path, disease_id, species_id):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return image, disease_id, species_id


train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels, train_species))
train_ds = train_ds.shuffle(buffer_size=len(train_paths), seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(decode_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(make_train_preprocess_fn(cross_ids, spot_ids, texture_ids), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels, val_species))
val_ds = val_ds.map(decode_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess_eval, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels, test_species))
test_ds = test_ds.map(decode_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(preprocess_eval, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

model, base_model = build_model(num_classes=num_classes, num_species=num_species)
disease_loss = make_class_balanced_focal_loss(alpha=alpha, gamma=GAMMA_FOCAL, label_smoothing=LABEL_SMOOTHING)

history_all = {}
for key in [
    "disease_output_accuracy",
    "val_disease_output_accuracy",
    "loss",
    "val_loss",
]:
    history_all[key] = []

callbacks = [
    EarlyStopping(monitor="val_disease_output_accuracy", mode="max", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ModelCheckpoint("cnn_model_targeted_best.keras", monitor="val_disease_output_accuracy", mode="max", save_best_only=True, verbose=1),
]

print("\nStage 1/3: Freeze backbone, train heads")
set_unfreeze_ratio(base_model, ratio=0.0)
compile_model(model, disease_loss=disease_loss, lr=PHASE1_LR)
h1 = model.fit(train_ds, validation_data=val_ds, epochs=PHASE1_EPOCHS, callbacks=callbacks, verbose=1)

print("\nStage 2/3: Unfreeze last 30% of backbone")
set_unfreeze_ratio(base_model, ratio=0.30)
compile_model(model, disease_loss=disease_loss, lr=PHASE2_LR)
h2 = model.fit(train_ds, validation_data=val_ds, epochs=PHASE2_EPOCHS, callbacks=callbacks, verbose=1)

print("\nStage 3/3: Unfreeze last 60% of backbone")
set_unfreeze_ratio(base_model, ratio=0.60)
compile_model(model, disease_loss=disease_loss, lr=PHASE3_LR)
h3 = model.fit(train_ds, validation_data=val_ds, epochs=PHASE3_EPOCHS, callbacks=callbacks, verbose=1)

for h in (h1, h2, h3):
    for k in history_all.keys():
        history_all[k].extend(h.history.get(k, []))

model.save(OUTPUT_MODEL_PATH)
print(f"Saved model: {OUTPUT_MODEL_PATH}")

# ===================== EVALUATION =====================
print("\nEvaluating on test split...")
preds = model.predict(test_ds, verbose=1)
if isinstance(preds, list):
    disease_probs = preds[0]
else:
    disease_probs = preds

y_pred = np.argmax(disease_probs, axis=1)
y_true = np.array(test_labels)

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("TARGETED TRAINING RECIPE - CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Targeted Recipe)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_CM_PLOT, dpi=300)
plt.close()
print(f"Saved confusion matrix: {OUTPUT_CM_PLOT}")

# Training curves
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history_all["disease_output_accuracy"], label="Train Disease Accuracy")
plt.plot(history_all["val_disease_output_accuracy"], label="Val Disease Accuracy")
plt.axvline(x=PHASE1_EPOCHS - 1, color="r", linestyle="--", label="Stage 1->2")
plt.axvline(x=PHASE1_EPOCHS + PHASE2_EPOCHS - 1, color="g", linestyle="--", label="Stage 2->3")
plt.title("Disease Accuracy by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_all["loss"], label="Train Loss")
plt.plot(history_all["val_loss"], label="Val Loss")
plt.axvline(x=PHASE1_EPOCHS - 1, color="r", linestyle="--", label="Stage 1->2")
plt.axvline(x=PHASE1_EPOCHS + PHASE2_EPOCHS - 1, color="g", linestyle="--", label="Stage 2->3")
plt.title("Loss by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_HISTORY_PLOT, dpi=300)
plt.close()
print(f"Saved training curves: {OUTPUT_HISTORY_PLOT}")

print("\nTargeted training pipeline complete.")
