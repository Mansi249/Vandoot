# Vandoot: Edge-AI Forest Threat Detection System 🌲🔥

An automated forest monitoring system leveraging **Edge AI** to detect wildfires, intruders, and animals in real-time. Designed to run on resource-constrained hardware (ESP32 + Raspberry Pi) using **Quantized MobileNetV1**.

### 🚀 Key Features
* **Edge Intelligence:** Deploys a custom Vision AI model directly on low-power devices.
* **Optimized Performance:** Achieved a **<300KB model size** using **Int8 Quantization**, enabling inference on limited RAM.
* **False Alarm Reduction:** Implemented "Context-Aware" training with diverse seasonal datasets (e.g., autumn foliage) to maintain **92% accuracy** without confusing warm colors for fire.
* **Distributed Architecture:** Orchestrates communication between ESP32 satellite nodes (Vision) and a Raspberry Pi Central Hub (Alerts).

### 🛠️ Tech Stack
* **Hardware:** Raspberry Pi 4, ESP32-CAM
* **AI/ML:** TensorFlow Lite, OpenCV, MobileNetV1
* **Language:** Python 3.9, C++ (Arduino IDE)
* **Optimization:** Post-training Quantization (Int8)

### 📊 Model Performance (The "Why")
Standard MobileNet models are too large (~15MB) for edge microcontrollers. 
* **My Approach:** I retrained MobileNetV1 on a curated dataset and applied post-training quantization.
* **Result:** Reduced model size by **~98%** (from 15MB to ~280KB) with minimal accuracy loss.

### 📂 Project Structure
* `src/train_model.py`: Training pipeline with data augmentation.
* `src/quantize.py`: Script used to convert .h5 model to .tflite.
* `models/`: Contains the final quantized .tflite model.

---
*Note: This was a collaborative group project. My primary contributions were the AI pipeline design, model quantization, and backend integration.*
