from tensorflow import keras

# Load the model
model = keras.models.load_model(".MODEL_NAME.keras")

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional quantization
tflite_model = converter.convert()

with open("MODEL_NAME.tflite", "wb") as f:
    f.write(tflite_model)
