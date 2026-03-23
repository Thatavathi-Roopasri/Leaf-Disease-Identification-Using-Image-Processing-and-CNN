import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("cnn_model.keras")
print(f"Total layers: {len(model.layers)}")
for i, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.Model):
        print(f"Layer {i} is a nested Model: {layer.name}")
        continue
    if layer.name.startswith("global_average_pooling"):
        print(f"Found global_average_pooling at index {i}")
        break
