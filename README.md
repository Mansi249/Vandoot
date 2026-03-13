# 🌲 VanDoot: Multi-Sensor Fusion & Distributed Gateway Logic

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Role](https://img.shields.io/badge/Role-Backend%20%26%20Data%20Architect-orange)
![Logic](https://img.shields.io/badge/Logic-Random%20Forest%20Judge-green)

VanDoot is an IoT infrastructure project designed to detect forest threats (Fire, Poachers) using a **Distributed Edge-Intelligence** model. 

This repository contains the **Central Gateway Logic**, which fuses data from multiple ESP32-CAM nodes to eliminate false positives and provide high-confidence alerts.

---

## 👥 My Role: Backend & Data Architect
*Focused on system scalability, data reliability, and sensor fusion logic.*

* **Decision Engine:** Designed a **Random Forest "Judge"** to resolve data conflicts between visual AI and environmental sensors (Smoke, Thermal, Acoustic).
* **Data Synthesis:** Built a custom **Synthetic Dataset Generator** to simulate 1,000+ forest threat scenarios, including sensor failures and environmental noise.
* **System Reliability:** Achieved a **1.00 F1-Score** in synthetic testing, proving the system can successfully override a "False Vision" (e.g., a red sunset) using Heat and Chemical data.

---

## 🏗️ Gateway Logic Flow (The Backend)

The Gateway (Raspberry Pi) acts as a micro-backend that processes data from multiple "Sentry" nodes using the following pipeline:

1.  **Ingestion:** (Planned) Listen for binary packets from distributed nodes.
2.  **Logic Processing:** Load the `vandoot_judge.pkl` engine to perform **Sensor Fusion**.
3.  **Conflict Resolution:**
    * **Vision says "Fire" + Smoke is Low** $\rightarrow$ **Verdict: SAFE** (Sunset/Reflection).
    * **Vision says "Human" + Audio is High** $\rightarrow$ **Verdict: POACHER** (Chainsaw/Logging).
    * **Vision says "Human" + Audio is Low** $\rightarrow$ **Verdict: SAFE** (Hiker/Shadow).

### **Decision Metrics (Current Model)**
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Safe (0)** | 1.00 | 1.00 | 1.00 |
| **Fire (1)** | 1.00 | 1.00 | 1.00 |
| **Human (2)**| 1.00 | 1.00 | 1.00 |

---

## 📂 System Components

* `synthetic_dataset_generator.py`: Logic to generate multi-modal sensor data.
* `random_forest.py`: The training pipeline for the fusion model.
* `vandoot_judge.pkl`: The compiled decision-making object (106KB).
* `verdict_simulator.py`: Interactive CLI to test gateway logic against manual inputs.

---

## 🔮 Scalability Roadmap
* [ ] **Binary Serialization:** Implementing `struct` packing to handle data ingestion from 100+ nodes.
* [ ] **Database Layer:** Integrate **SQLite/InfluxDB** for long-term threat trend analysis.
* [ ] **Async Processing:** Implement `asyncio` to handle concurrent LoRa packet arrivals.

---
*Note: This repository focuses on the Backend Logic and Sensor Fusion layer.*
