# 🚦 Adaptive Traffic Signal Control Simulation

An AI-based **Adaptive Traffic Signal Control System** using **SUMO, LSTM, GNN, and MAPPO**, with dedicated **Emergency Vehicle and VIP Priority Management**.

The system simulates **16 signalized intersections** and dynamically controls traffic signals based on traffic conditions.

---

# 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [System Architecture](#-system-architecture)
4. [AI Components](#-what-each-ai-component-does)
5. [Repository Structure](#-repository-structure)
6. [Important Folders and Files](#-important-folders-and-files)
7. [Requirements](#-requirements)
8. [Installation](#-installation)
9. [Running SUMO](#-running-sumo)
10. [Data Collection](#-data-collection)
11. [LSTM Prediction](#-lstm-prediction)
12. [GNN](#-gnn)
13. [MAPPO](#-mappo)
14. [Emergency & VIP System](#-emergency--vip-system)
15. [Evaluation](#-evaluation)
16. [Final Results](#-final-results)
17. [Visualization](#-visualization)
18. [Complete Execution Order](#-complete-execution-order)
19. [Existing Trained Models](#-existing-trained-models)
20. [Important Output Files](#-important-output-files)
21. [Troubleshooting](#-troubleshooting)
22. [Project Summary](#-project-summary)

---

# 📖 Project Overview

Traditional fixed-time traffic signals cannot effectively respond to continuously changing traffic conditions.

This project develops an **adaptive traffic signal control system** that combines traffic simulation, deep learning, graph neural networks, and multi-agent reinforcement learning.

The system uses:

* **LSTM** for temporal traffic prediction
* **GNN** for spatial relationships between intersections
* **MAPPO** for adaptive traffic-signal decisions
* **Emergency/VIP modules** for priority handling

The traffic network contains **16 signalized intersections**.

---

# ⭐ Features

* 🚦 16-intersection traffic simulation
* 🧠 LSTM-based traffic prediction
* 🕸️ GNN-based spatial traffic analysis
* 🤖 MAPPO-based multi-agent signal control
* 🚑 Emergency vehicle priority
* ⭐ VIP vehicle priority
* 📊 Fixed-time vs AI-based performance evaluation
* 💾 Pre-trained models included
* 🔄 Complete data collection, training and evaluation pipeline

---

# 🏗️ System Architecture

```text
                    SUMO
                     │
                     ▼
              Traffic Data
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
        LSTM                  GNN
   Traffic Prediction    Spatial Features
          │                     │
          └──────────┬──────────┘
                     ▼
                   MAPPO
                     │
              Signal Actions
                     │
                     ▼
             Emergency / VIP
                 Priority
                     │
                     ▼
                    SUMO
                     │
                     ▼
                Evaluation
                     │
                     ▼
                Final Results
```

---

# 🧠 What Each AI Component Does

| Component            | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| **SUMO**             | Simulates roads, vehicles and traffic signals          |
| **TraCI**            | Connects Python with SUMO                              |
| **LSTM**             | Predicts future traffic conditions                     |
| **GNN**              | Learns relationships between neighboring intersections |
| **MAPPO**            | Decides traffic-signal actions                         |
| **Emergency Module** | Detects and prioritizes emergency vehicles             |
| **VIP Module**       | Provides lower-priority control for VIP vehicles       |

Priority hierarchy:

```text
Emergency > VIP > Normal Traffic
```

---

# 📁 Repository Structure

```text
Adaptive-Traffic-Control/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── config.yaml
│   ├── network.yaml
│   └── training.yaml
│
├── sumo/
│   ├── network/
│   ├── routes/
│   ├── simulation/
│   └── tls/
│
├── sumo-rl/
│   └── sumo_rl/nets/RESCO/grid4x4/
│
├── src/
│   ├── environment/
│   ├── data/
│   ├── prediction/
│   ├── graph/
│   ├── agents/
│   ├── emergency/
│   ├── evaluation/
│   └── final_visualizer.py
│
├── training/
│   ├── train_lstm.py
│   ├── train_gnn.py
│   └── train_mappo.py
│
├── models/
│   ├── lstm/
│   ├── gnn/
│   └── mappo/
│
├── evaluation/
│   ├── results/
│   ├── plots/
│   └── tables/
│
└── scripts/
```

---

# 📂 Important Folders and Files

## `configs/`

Contains project configuration.

* `config.yaml` → General configuration
* `network.yaml` → Network settings
* `training.yaml` → Training parameters

---

## `sumo/`

Contains the project's SUMO configuration.

```text
network/       → Road and intersection network
routes/        → Normal/emergency routes
simulation/    → SUMO simulation configuration
tls/           → Traffic-light configuration
```

Main configuration:

```text
sumo/simulation/simulation.sumocfg
```

---

## `sumo-rl/`

Contains the RESCO `grid4x4` traffic network.

```text
sumo-rl/
└── sumo_rl/
    └── nets/
        └── RESCO/
            └── grid4x4/
```

The network contains **16 signalized intersections**.

---

# 🧩 `src/`

Contains the main implementation.

### `src/environment/`

Handles SUMO communication and traffic-signal control.

```text
sumo_env.py    → SUMO environment / TraCI
state.py       → Traffic state
action.py      → Signal actions
reward.py      → Reinforcement-learning reward
```

### `src/data/`

Handles traffic data.

```text
collector.py       → Collects traffic data
dataset.py         → Creates datasets
preprocessing.py   → Cleans/normalizes data
```

Traffic features include:

```text
Vehicle Count
Waiting Time
Average Speed
Queue Length
```

### `src/prediction/`

Handles LSTM prediction.

```text
lstm.py       → LSTM model
predict.py    → Generates predictions
```

### `src/graph/`

Handles graph processing.

```text
graph_builder.py    → Builds intersection graph
graph_dataset.py    → Prepares graph data
gnn.py              → GNN model
```

### `src/agents/`

Contains MAPPO implementation.

```text
actor.py      → Selects actions
critic.py     → Evaluates states
buffer.py     → Stores experiences
mappo.py      → MAPPO implementation
```

### `src/emergency/`

Handles emergency and VIP vehicles.

```text
detector.py
priority.py
route.py
route_generator.py
scenario_generator.py
scenario_routes.py
```

### `src/evaluation/`

Contains evaluation and comparison scripts.

```text
metrics.py
evaluate.py
comparison.py
final_results.py
final_emergency_comparison.py
```

### `src/final_visualizer.py`

Generates final project visualizations automatically.

---

# 💻 Requirements

* Windows 10/11
* Python 3.10+
* SUMO 1.27.1 recommended
* Git
* VS Code recommended

---

# ⚙️ Installation

## 1. Clone Repository

```powershell
git clone https://github.com/Harshul055/Adaptive-Traffic-Signal-Control-Simulation.git
cd Adaptive-Traffic-Signal-Control-Simulation
```

## 2. Create Virtual Environment

```powershell
python -m venv venv
```

Activate:

```powershell
.\venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 4. Verify Python

```powershell
python --version
```

## 5. Verify SUMO

```powershell
sumo-gui --version
```

If SUMO is not recognized, add its `bin` directory to Windows PATH.

Typical location:

```text
C:\Program Files (x86)\Eclipse\Sumo\bin
```

---

# ▶️ Running SUMO

Run the simulation:

```powershell
sumo-gui -c sumo/simulation/simulation.sumocfg
```

SUMO-GUI should display the road network, vehicles and traffic signals.

---

# 📊 Data Collection

Collect traffic data from SUMO:

```powershell
python scripts/collect_data.py
```

Output:

```text
evaluation/results/traffic_data.csv
```

Collected features:

* Vehicle count
* Waiting time
* Average speed
* Queue length

---

# 🔮 LSTM Prediction

Train the LSTM model:

```powershell
python training/train_lstm.py
```

Model:

```text
models/lstm/traffic_lstm.pth
```

Generate predictions:

```powershell
python src/prediction/predict.py
```

Output:

```text
evaluation/results/lstm_predictions.csv
```

---

# 🕸️ GNN

The GNN represents the 16 intersections as graph nodes and learns relationships between neighboring intersections.

Train the GNN:

```powershell
python training/train_gnn.py
```

Model:

```text
models/gnn/traffic_gnn.pth
```

---

# 🤖 MAPPO

MAPPO treats each traffic-light intersection as an independent agent.

For this project:

```text
16 Intersections
       ↓
16 MAPPO Agents
```

Train MAPPO:

```powershell
python training/train_mappo.py
```

Models are saved in:

```text
models/mappo/
```

---

# 🚑 Emergency & VIP System

The system detects:

```text
Emergency
Ambulance
Fire
Police
VIP
```

Priority hierarchy:

```text
Emergency
    ↓
VIP
    ↓
Normal Traffic
```

Emergency vehicles can override normal signal-control decisions when priority is required.

---

# 🚨 Emergency/VIP Scenario

The final scenario contains:

```text
5 Emergency Vehicles
+
5 VIP Vehicles
=
10 Special Vehicles
```

Generate the scenario:

```powershell
python scripts/merge_traffic_scenario.py
```

Test detection:

```powershell
python scripts/test_emergency_detection.py
```

---

# 📈 Evaluation

The project compares:

```text
Fixed-Time Control
        VS
GNN + MAPPO + Emergency/VIP
```

Main metrics:

| Metric        | Desired Direction |
| ------------- | ----------------- |
| Vehicle Count | Lower             |
| Waiting Time  | Lower             |
| Average Speed | Higher            |
| Queue Length  | Lower             |

Run evaluation:

```powershell
python scripts/run_mappo_evaluation.py
```

---

# 🏆 Final Results

| Metric        | Fixed-Time | GNN + MAPPO + Emergency/VIP |       Improvement |
| ------------- | ---------: | --------------------------: | ----------------: |
| Vehicle Count |      68.11 |                       32.12 |  **52.85% lower** |
| Waiting Time  |    2136.66 |                      108.95 |  **94.90% lower** |
| Average Speed |   6.72 m/s |                   13.14 m/s | **95.51% higher** |
| Queue Length  |      26.67 |                        8.78 |  **67.06% lower** |

These results are from the project's tested simulation scenario.

---

# 📊 Visualization

Generate the final plots:

```powershell
python src/final_visualizer.py
```

Output directory:

```text
evaluation/plots/final/
```

The visualization pipeline includes:

* Traffic metric comparisons
* Improvement analysis
* Time-series analysis
* Traffic distributions
* Correlation analysis
* Final results tables

---

# 🔄 Complete Execution Order

For a complete reproduction from scratch:

```text
1. Install Python + SUMO
          ↓
2. Install requirements
          ↓
3. Test SUMO
          ↓
4. Collect traffic data
          ↓
5. Train LSTM
          ↓
6. Generate LSTM predictions
          ↓
7. Train GNN
          ↓
8. Generate Emergency/VIP scenario
          ↓
9. Train MAPPO
          ↓
10. Run MAPPO evaluation
          ↓
11. Compare with Fixed-Time
          ↓
12. Generate final visualizations
```

Commands:

```powershell
python scripts/collect_data.py
python training/train_lstm.py
python src/prediction/predict.py
python training/train_gnn.py
python scripts/merge_traffic_scenario.py
python training/train_mappo.py
python scripts/run_mappo_evaluation.py
python scripts/evaluate.py
python src/final_visualizer.py
```

---

# 💾 Existing Trained Models

Pre-trained models are already included:

```text
models/
├── lstm/
│   └── traffic_lstm.pth
│
├── gnn/
│   └── traffic_gnn.pth
│
└── mappo/
    ├── actor_01.pth ... actor_16.pth
    └── critic_01.pth ... critic_16.pth
```

Therefore, retraining is not required simply to evaluate the existing trained system.

---

# 📌 Important Output Files

| File                             | Purpose                |
| -------------------------------- | ---------------------- |
| `traffic_data.csv`               | Collected traffic data |
| `lstm_predictions.csv`           | LSTM predictions       |
| `fixed_time_results.csv`         | Fixed-time baseline    |
| `mappo_final_results.csv`        | Final MAPPO results    |
| `final_emergency_comparison.csv` | Final comparison       |
| `final_results.csv`              | Final results table    |

---

# 🔧 Troubleshooting

### `No module named 'src'`

Run commands from the project root:

```powershell
cd C:\Project\Adaptive-Traffic-Signal-Control-Simulation
```

For module-based tests:

```powershell
python -m src.agents.test_mappo
```

### SUMO not found

Check:

```powershell
sumo-gui --version
```

If necessary, add:

```text
C:\Program Files (x86)\Eclipse\Sumo\bin
```

to PATH.

### Missing Python packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 👨‍💻 Project Summary

This project combines:

```text
SUMO
 +
TraCI
 +
LSTM
 +
GNN
 +
MAPPO
 +
Emergency Priority
 +
VIP Priority
```

to create an adaptive multi-intersection traffic signal control system that responds to changing traffic conditions and provides priority handling for special vehicles.
﻿
