# Vandoot: Edge-AI Forest Threat Detection System 🌲🔥
> **Current Status:** Phase 1 Complete (AI Model Design, Training & Validation).  
> **Performance:** 92% Accuracy on Test Set | <300KB Model Size.

Vandoot is a distributed IoT system designed to detect forest fires, intruders, and wildlife in real-time. This repository currently hosts the **Computer Vision & Optimization Pipeline** (The "Brain"), engineered to run on ultra-low-power ESP32 microcontrollers using quantized TensorFlow Lite.

### 🎯 Project Goal
To replace expensive industrial sensors with a low-cost network of **ESP32-CAM nodes** (Satellite) and a **Raspberry Pi** (Central Hub), using on-device Edge AI to minimize bandwidth and power consumption.

---

### ✅ Phase 1 Achievements (Software & AI)
The focus of this phase was to architect a vision model small enough to run on a microcontroller (<500KB) without losing accuracy.

- [x] **Dataset Synthesis:** Curated and pre-processed 2,000+ images (Fire vs. Non-Fire) resized to 96x96 for embedded compatibility.
- [x] **Model Architecture:** Retrained **MobileNetV1 (Alpha 0.25)** for high-speed inference.
- [x] **Int8 Quantization:** Successfully compressed the model from **~15MB (Float32)** to **<300KB (Int8)** using TensorFlow Lite Post-Training Quantization.
- [x] **Validation:** Achieved 97% Precision on Fire detection, ensuring minimal false alarms.

### 📊 Model Performance
I trained and quantized the model specifically for this task. Below are the actual evaluation results on the test dataset (2,039 images):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **🔥 Fire** | **0.97** | 0.88 | 0.92 | 523 |
| **👤 Human** | 0.86 | **0.99** | 0.92 | 503 |
| **🐾 Animal** | 0.97 | 0.94 | 0.95 | 480 |
| **🌫️ Empty** | 0.89 | 0.87 | 0.88 | 533 |

**Key Takeaways:**
* **Reliability:** The model has **97% Precision for Fire**, meaning false positives (confusing a red shirt for fire) are extremely rare.
* **Safety:** The **99% Recall for Humans** ensures that if a poacher or intruder is present, the system almost never misses them.

---

### 📂 Repository Structure
Since hardware integration is pending, this repo focuses on the ML pipeline:

```text
Vandoot/
├── 📁 datasets/            # Training data (Fire vs Non-Fire)
├── 📁 models/              # The Trained "Brain"
│   ├── vandoot_model.tflite     # Final Quantized Model
│   └── model_data.cc            # C-array Hex Dump for ESP32
├── 📁 src/
│   ├── preprocess_dataset.py  # Script 1: Resizes raw images to 96x96
│   ├── train_model.py         # Script 2: Trains MobileNetV1 & converts to Int8
│   └── evaluate_model.py      # Script 3: Generates Accuracy Report & Confusion Matrix
└── requirements.txt
🛠️ Tech Stack
Core: Python 3.9, TensorFlow 2.x

Edge ML: TensorFlow Lite Micro (Int8 Quantization)

Data Processing: OpenCV, NumPy

Hardware Target: ESP32-CAM (AI Node), Raspberry Pi 4 (Gateway)

🚀 How to Replicate Results
Clone the repo:

Bash

git clone [https://github.com/Mansi249/Vandoot-Edge-AI.git](https://github.com/Mansi249/Vandoot-Edge-AI.git)
cd Vandoot-Edge-AI
Install dependencies:

Bash

pip install -r requirements.txt
Run the pipeline:

Step 1: python src/preprocess_dataset.py (Prepares images)

Step 2: python src/train_model.py (Trains & creates .tflite file)

Step 3: python src/evaluate_model.py (Shows accuracy graphs)

Developed by Mansi Sangwan. This project is part of an ongoing R&D initiative to democratize forest monitoring technology. Note: This was a collaborative group project. My primary contributions were the AI pipeline design, model quantization, and backend integration.