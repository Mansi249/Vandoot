# 🌲 Vandoot: Edge-AI Forest Threat Detection

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Model](https://img.shields.io/badge/Model-MobileNetV1%20(Quantized)-green)
![Status](https://img.shields.io/badge/Status-Prototype%20(Phase%201)-yellow)

An IoT & Computer Vision project designed to detect **Forest Fires, Intruders, and Wildlife** in real-time. This repository hosts the **AI & Software Pipeline**, currently optimized for ESP32 deployment.

---

## 👥 Project Context & Roles
**This is a collaborative Group Project developed by a team of engineering students.**

* **My Role (Lead AI Engineer):**
    * Architected the **Computer Vision Pipeline** (MobileNetV1).
    * Executed **Dataset Synthesis** and pre-processing.
    * Performed **Int8 Quantization** to fit the model into <300KB flash memory.
    * Generated the **C-Byte Array** firmware integration for the ESP32.
* **Team Scope:** Hardware assembly, power management (Solar/Battery), and physical casing design are handled by other team members.

---

## 📖 Project Overview

Traditional forest monitoring relies on slow satellite data or expensive sensors. This project builds a **Distributed Intelligence** system using "TinyML" to detect threats in milliseconds directly on the edge.

**Current Focus:**
* **Phase 1 (Completed):** Vision AI design, training, and quantization.
* **Phase 2 (In Progress):** Audio analysis, Sensor Fusion, and Hardware integration.

## 📂 Dataset & Constraints

* **Dataset:** ~2,000 synthesized images (Fire, Human, Animal, Empty) resized to **96x96 pixels**.
* **Target Hardware:** ESP32-CAM (AI Node) + Raspberry Pi 4 (Gateway).
* **Constraint:** Model compressed to **<300KB (Int8)** to fit within microcontroller flash memory.

## 🛠️ Methodology (My Contribution)

1.  **Data Pipeline:** Automated resizing script (`preprocess_dataset.py`) to standardize inputs.
2.  **Model Engineering:**
    * Retrained **MobileNetV1 (Alpha 0.25)** for high-speed inference.
    * Applied **Post-Training Quantization (Int8)** using TensorFlow Lite.
3.  **Simulation:** Verified logic with a custom Python inference script before hardware deployment.

## 📊 Phase 1 Performance (The "Brain")

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Model Size** | **280 KB** | Reduced from 15MB (98% compression). |
| **Accuracy** | **92%** | Tested on 2,039 validation images. |
| **Fire Precision**| **0.97** | Ultra-low false alarm rate. |

## 🚀 How to Run (Simulation)

1.  **Clone & Install:**
    ```bash
    git clone [https://github.com/Mansi249/Vandoot-Edge-AI.git](https://github.com/Mansi249/Vandoot-Edge-AI.git)
    pip install -r requirements.txt
    ```
2.  **Run the AI Pipeline:**
    ```bash
    python src/train_model.py  # Generates the .tflite model
    python src/evaluate_model.py  # Displays accuracy graphs
    ```

## 🔮 Roadmap & Future Work

The AI "Brain" is ready. The next steps focus on physical deployment:
* [ ] **Hardware:** Flash the C-byte array onto ESP32-CAM units.
* [ ] **Comms:** Implement LoRa (Long Range) protocol for forest-wide data transmission.
* [ ] **Power:** Optimize deep-sleep cycles for solar-powered operation.

---
*Note: This repository contains the source code for the AI/Software subsystem only.*
