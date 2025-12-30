
import numpy as np
import tensorflow as tf
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "vandoot_model.tflite"
DATASET_PATH = "datasets"
IMG_SIZE = 96
CLASSES = ['0_fire', '1_human', '2_animal', '3_empty']

# ==========================================
# 1. LOAD TFLITE MODEL
# ==========================================
print(f"[1] Loading TFLite Model: {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check if model expects Quantized (int8) or Float inputs
input_type = input_details[0]['dtype']
print(f"Model Input Type: {input_type}")

# ==========================================
# 2. RUN INFERENCE ON ALL IMAGES
# ==========================================
print("\n[2] Running Inference (be patient)...")

y_true = []
y_pred = []

for class_index, class_name in enumerate(CLASSES):
    folder_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.exists(folder_path):
        print(f"Skipping missing folder: {folder_path}")
        continue
        
    files = os.listdir(folder_path)
    print(f"Processing {class_name} ({len(files)} images)...")
    
    for file in files:
        img_path = os.path.join(folder_path, file)
        
        # Load and Preprocess Image
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # TF expects RGB
        
        # Handle Input Data Type
        if input_type == np.float32:
            # If float, scale to [0, 1]
            input_data = (img.astype(np.float32) / 255.0)
        else:
            # If uint8 (Quantized), keep [0, 255]
            input_data = img.astype(input_type)
            
        # Add batch dimension (1, 96, 96, 3)
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run Model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get Prediction
        predicted_index = np.argmax(output_data)
        
        y_true.append(class_index)
        y_pred.append(predicted_index)

# ==========================================
# 3. GENERATE REPORT
# ==========================================
print("\n" + "="*40)
print("FINAL EVALUATION REPORT")
print("="*40)

# Precision, Recall, F1-Score
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted by AI')
plt.ylabel('Actual Label')
plt.title('VanDoot AI Confusion Matrix')
plt.show()