import tensorflow as tf
import cv2
import numpy as np
import sys

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "vandoot_model.tflite"
IMG_SIZE = 96
CLASSES = ['0_fire', '1_human', '2_animal', '3_empty']

# CHANGE THIS to your specific image path!
# Example: "C:/Users/singh/Desktop/test_fire.jpg"
IMAGE_PATH = "C:\VANDOOT\download (3).jpg" 

# ==========================================
# RUN INFERENCE
# ==========================================
# 1. Load Model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Load & Preprocess Image
print(f"Testing Image: {IMAGE_PATH}")
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("ERROR: Could not read image. Check the path!")
    sys.exit()

# Resize to match training (96x96)
img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# Prepare Input Data
input_type = input_details[0]['dtype']
if input_type == np.float32:
    input_data = (img_rgb.astype(np.float32) / 255.0)
else:
    input_data = img_rgb.astype(input_type)

input_data = np.expand_dims(input_data, axis=0)

# 3. Predict
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# 4. Show Results
print("\n--- AI PREDICTION ---")
predictions = output_data[0]

# Handle Quantized Output (0-255) vs Float Output (0.0-1.0)
if input_type != np.float32:
    predictions = predictions / 255.0  # Normalize to 0-1

for i, score in enumerate(predictions):
    bar = "█" * int(score * 20)
    print(f"{CLASSES[i]:<10}: {score:.4f}  {bar}")

winner_index = np.argmax(predictions)
print(f"\n🏆 RESULT: {CLASSES[winner_index].upper()} ({predictions[winner_index]*100:.1f}%)")