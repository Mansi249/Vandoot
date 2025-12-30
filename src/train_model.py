
import tensorflow as tf
import numpy as np
import os
import pathlib

# ==========================================
# CONFIGURATION (Do not change for ESP32)
# ==========================================
DATASET_PATH = "VanDoot_Ready_96x96"  # Folder from Day 1
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20  # Increase to 50 if accuracy is low
MODEL_ALPHA = 0.25  # 0.25 = MobileNet-Tiny (Critical for ESP32 speed)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Checking for GPU: {tf.config.list_physical_devices('GPU')}")

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("\n[1] Loading Dataset...")
data_dir = pathlib.Path(DATASET_PATH)

# Split: 80% Training, 20% Validation
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Classes found: {class_names}")
# Expected: ['0_fire', '1_human', '2_animal', '3_empty']

# Optimize performance (Prefetching)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 2. BUILD MODEL (MobileNetV1 Tiny)
# ==========================================
print("\n[2] Building MobileNetV1 (Alpha 0.25)...")

# Data Augmentation (Helps with small datasets)
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.1),
  tf.keras.layers.RandomZoom(0.1),
])

# Pre-processing: Rescale pixel values from [0-255] to [0-1]
rescale = tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Base Model (Transfer Learning)
base_model = tf.keras.applications.MobileNet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=MODEL_ALPHA,  # Makes it "Tiny"
    include_top=False,  # We remove the ImageNet classifier
    weights='imagenet'
)
base_model.trainable = False # Freeze base for first pass

model = tf.keras.Sequential([
    rescale,
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 3. TRAINING
# ==========================================
print("\n[3] Starting Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==========================================
# 4. CONVERT TO TFLITE (QUANTIZATION)
# ==========================================
print("\n[4] Converting to TFLite for ESP32...")

# Representative Dataset Generator (Needed for INT8 Quantization)
# This teaches the converter the range of values in your images
def representative_data_gen():
    for input_value, _ in train_ds.take(100):
        # Model expects float32 [0, 255] which Rescaling layer handles, 
        # but TFLite converter needs raw input.
        yield [tf.cast(input_value, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure strict INT8 operations for ESP32
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # Input is image (0-255)
converter.inference_output_type = tf.int8  # Output is probability

tflite_model = converter.convert()

# Save the file
with open('vandoot_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"\n[SUCCESS] Model saved: vandoot_model.tflite")
print(f"Size: {len(tflite_model) / 1024:.2f} KB (Target: < 500 KB)")

# ==========================================
# 5. GENERATE C-BYTE ARRAY (Hex Dump)
# ==========================================
print("\n[5] Generating C Header file...")

def hex_to_c_array(model_data, model_name="vandoot_model"):
    c_str = ""
    c_str += "#include <cstdint>\n\n"
    c_str += f"unsigned int {model_name}_len = {len(model_data)};\n"
    c_str += f"unsigned char {model_name}[] = {{\n"
    
    hex_array = []
    for i, val in enumerate(model_data):
        hex_array.append(f"0x{val:02x}")
        if (i + 1) % 12 == 0:
            c_str += "  " + ", ".join(hex_array) + ",\n"
            hex_array = []
    
    if hex_array:
        c_str += "  " + ", ".join(hex_array) + "\n"
        
    c_str += "};\n"
    return c_str

with open('model_data.cc', 'w') as f:
    f.write(hex_to_c_array(tflite_model))

print("[DONE] Saved 'model_data.cc'. This file goes into your ESP32 Arduino Sketch.")