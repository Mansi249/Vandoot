# Vandoot: Edge-AI Forest Threat Detection System 🌲🔥
> **Current Status:** Phase 1 Complete (AI Model Design, Training & Quantization).  
> **Next Phase:** Hardware Integration (ESP32/LoRa) & Field Testing.

Vandoot is a distributed IoT system designed to detect forest fires and intruders in real-time. This repository currently hosts the **Computer Vision & Optimization Pipeline** (The "Brain"), engineered to run on ultra-low-power edge devices.

### 🎯 Project Goal
To replace expensive industrial sensors with a low-cost, distributed network of **ESP32-CAM nodes** (Satellite) and a **Raspberry Pi** (Central Hub), using on-device Edge AI to minimize bandwidth and power.

---

### ✅ Phase 1 Achievements (Software & AI)
The focus of this phase was to architect a vision model small enough to run on a microcontroller (<500KB) without losing accuracy.

- [x] **Dataset Synthesis:** Curated and pre-processed 1,000+ images (Fire vs. Non-Fire) resized to 96x96 for embedded compatibility.
- [x] **Model Architecture:** Retrained **MobileNetV1 (Alpha 0.25)** for high-speed inference.
- [x] **Int8 Quantization:** Successfully compressed the model from **~15MB (Float32)** to **<300KB (Int8)** using TensorFlow Lite Post-Training Quantization.
- [x] **Simulation:** Verified inference logic in Python before hardware deployment.

### 🚧 Phase 2 Roadmap (Hardware Implementation)
- [ ] Flash `.cc` model array to **ESP32-CAM**.
- [ ] Implement **Zero Crossing Rate (ZCR)** algorithm for basic audio anomaly detection (e.g., Chainsaws).
- [ ] Establish **LoRa (Long Range)** communication protocol between Nodes and Hub.
- [ ] Deploy **Random Forest** validation logic on Raspberry Pi.

---

### 📂 Repository Structure
Since hardware integration is pending, this repo focuses on the ML pipeline:

```text
Vandoot/
├── 📁 datasets/            # Scripts used for dataset synthesis & resizing
├── 📁 models/              # The Trained "Brain"
│   ├── model_float32.tflite     # Original Model (High Accuracy)
│   └── model_quantized.cc       # FINAL OUTPUT: C-array for ESP32 (<300KB)
├── 📁 src/
│   ├── train_mobilenet.py  # Transfer Learning script
│   ├── quantize_model.py   # Optimization script (Float32 -> Int8)
│   └── simulate_inference.py # PC-based testing script
└── requirements.txt
