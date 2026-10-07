# 🚦 Adaptive Traffic Signal Control Simulation

An AI-based Adaptive Traffic Signal Control System using **SUMO, TraCI, LSTM, GNN, and MAPPO**, with dedicated **Emergency Vehicle and VIP Priority Management**. The LSTM provides temporal traffic prediction as a separate module; the matched paper evaluator currently measures **GNN + MAPPO + Emergency/VIP**.

The system simulates a **4×4 traffic network containing 16 signalized intersections** and evaluates adaptive AI-based traffic-signal control against a fixed-time baseline.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [System Architecture](#-system-architecture)
4. [AI Components](#-ai-components)
5. [Repository Structure](#-repository-structure)
6. [Understanding the Project Root](#-understanding-the-project-root)
7. [Important Folders and Files](#-important-folders-and-files)
8. [Recommended Development Environment](#-recommended-development-environment)
9. [Requirements](#-requirements)
10. [Installation](#-installation)
11. [SUMO Installation and PATH Setup](#-sumo-installation-and-path-setup)
12. [Running the Project](#-running-the-project)
13. [Using app.py](#-using-apppy)
14. [Data Collection](#-data-collection)
15. [LSTM Training](#-lstm-training)
16. [GNN Training](#-gnn-training)
17. [MAPPO Training](#-mappo-training)
18. [Emergency and VIP System](#-emergency-and-vip-system)
19. [Evaluation](#-evaluation)
20. [Final Results](#-final-results)
21. [Output Files](#-output-files)
22. [Complete Execution Order](#-complete-execution-order)
23. [Existing Trained Models](#-existing-trained-models)
24. [Troubleshooting](#-troubleshooting)
25. [Project Summary](#-project-summary)

---

# 📖 Project Overview

Traditional fixed-time traffic signals use predefined timing plans and cannot easily react to continuously changing traffic conditions.

This project develops an **AI-based adaptive traffic signal control system** that combines:

* **SUMO** for traffic simulation
* **TraCI** for Python ↔ SUMO communication
* **LSTM** for temporal traffic prediction
* **GNN** for spatial relationships between intersections
* **MAPPO** for multi-agent traffic-signal control
* **Emergency Priority** for emergency vehicles
* **VIP Priority** for VIP vehicles
* **Evaluation tools** for comparing AI control with fixed-time control

The simulated network contains **16 signalized intersections**.

The main objective is to reduce:

* Waiting time
* Queue length
* Traffic congestion

while improving:

* Average vehicle speed
* Overall traffic flow

---

#  Features

*  16-intersection traffic simulation
*  LSTM-based traffic prediction
*  GNN-based spatial traffic analysis
*  MAPPO-based multi-agent signal control
*  Emergency vehicle priority
*  VIP vehicle priority
*  Fixed-time vs AI-based comparison
*  Pre-trained LSTM, GNN and MAPPO models
*  Automated result generation
*  Traffic performance analysis
*  Windows + VS Code workflow
*  `app.py` main project launcher

---

#  System Architecture

```text
                         SUMO
                           │
                           ▼
                    Traffic Conditions
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
           LSTM                         GNN
    Temporal Prediction          Spatial Relationships
             │                           │
             └─────────────┬─────────────┘
                           ▼
                         MAPPO
                  Multi-Agent Control
                           │
                           ▼
                  Traffic Signal Actions
                           │
                           ▼
                 Emergency / VIP Priority
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

#  AI Components

| Component        | Purpose                                                    |
| ---------------- | ---------------------------------------------------------- |
| SUMO             | Simulates roads, vehicles and traffic signals              |
| TraCI            | Connects Python with SUMO                                  |
| LSTM             | Predicts future traffic conditions                         |
| GNN              | Represents relationships between neighboring intersections |
| MAPPO            | Controls traffic signals using multiple agents             |
| Emergency Module | Detects and prioritizes emergency vehicles                 |
| VIP Module       | Provides priority handling for VIP vehicles                |

### Priority hierarchy

```text
Emergency
    ↓
VIP
    ↓
Normal Traffic
```

Emergency vehicles receive the highest priority.

---

# 📁 Repository Structure

```text
Adaptive-Traffic-Control/
│
├── README.md
├── app.py
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
│   │   ├── network.net.xml
│   │   ├── nodes.nod.xml
│   │   └── edges.edg.xml
│   │
│   ├── routes/
│   │   ├── traffic.rou.xml
│   │   └── emergency.rou.xml
│   │
│   ├── simulation/
│   │   └── simulation.sumocfg
│   │
│   └── tls/
│       └── traffic_lights.add.xml
│
├── sumo-rl/
│   └── sumo_rl/
│       └── nets/
│           └── RESCO/
│               └── grid4x4/
│
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── sumo_env.py
│   │   ├── state.py
│   │   ├── action.py
│   │   ├── reward.py
│   │   └── test_sumo_env.py
│   │
│   ├── data/
│   │   ├── collector.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   │
│   ├── prediction/
│   │   ├── lstm.py
│   │   └── predict.py
│   │
│   ├── graph/
│   │   ├── graph_builder.py
│   │   ├── graph_dataset.py
│   │   └── gnn.py
│   │
│   ├── agents/
│   │   ├── actor.py
│   │   ├── critic.py
│   │   ├── buffer.py
│   │   └── mappo.py
│   │
│   ├── emergency/
│   │   ├── detector.py
│   │   ├── priority.py
│   │   ├── route.py
│   │   ├── route_generator.py
│   │   ├── scenario_generator.py
│   │   └── scenario_routes.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   ├── comparison.py
│   │   ├── final_results.py
│   │   ├── final_emergency_comparison.py
│   │   ├── plot_results.py
│   │   ├── plot_lstm.py
│   │   └── plot_control.py
│   │
│   └── final_visualizer.py
│
├── training/
│   ├── train_lstm.py
│   ├── train_gnn.py
│   └── train_mappo.py
│
├── models/
│   ├── lstm/
│   │   └── traffic_lstm.pth
│   │
│   ├── gnn/
│   │   └── traffic_gnn.pth
│   │
│   └── mappo/
│       ├── actor_01.pth
│       ├── ...
│       ├── actor_16.pth
│       ├── critic_01.pth
│       └── ...
│
├── evaluation/
│   ├── results/
│   ├── plots/
│   └── tables/
│
└── scripts/
    ├── run_sumo.py
    ├── collect_data.py
    ├── train.py
    ├── evaluate.py
    ├── generate_emergency_scenario.py
    ├── merge_traffic_scenario.py
    ├── run_mappo_evaluation.py
    ├── run_scenario.py
    └── test_emergency_detection.py
```

---

# 📂 Understanding the Project Root

The **project root** means the main folder containing `README.md`, `app.py`, `requirements.txt`, `src`, `scripts`, `training`, etc.

For example, your current project can be located at:

```text
C:\Project\Adaptive-Traffic-Control
```

Therefore:

```text
<PROJECT_ROOT>
```

in this README means:

```text
C:\Project\Adaptive-Traffic-Control
```

for your computer.

Another user can place the project somewhere completely different, for example:

```text
D:\Projects\Adaptive-Traffic-Control
```

or:

```text
C:\Users\User\Desktop\Adaptive-Traffic-Control
```

The project does **not** require a specific drive or folder.

### Important

Commands should normally be executed from:

```text
<PROJECT_ROOT>
```

For your computer:

```powershell
cd "C:\Project\Adaptive-Traffic-Control"
```

Then commands such as:

```powershell
python app.py
```

will work from the correct location.

---

#  What Each Main Directory Stores

| Directory     | Stores                                             |
| ------------- | -------------------------------------------------- |
| `configs/`    | Configuration files                                |
| `sumo/`       | Main SUMO network, routes and signal configuration |
| `sumo-rl/`    | RESCO/grid4x4 reference network                    |
| `src/`        | Main Python implementation                         |
| `training/`   | Model training programs                            |
| `models/`     | Trained AI models                                  |
| `evaluation/` | CSV results, tables and plots                      |
| `scripts/`    | Project execution and testing scripts              |
| Project root  | README, `app.py`, requirements and Git files       |

---

#  Important Source Directories

## `src/environment/`

Controls the SUMO environment.

```text
sumo_env.py
```

Handles the SUMO/TraCI environment.

```text
state.py
```

Handles traffic-state information.

```text
action.py
```

Handles traffic-signal actions.

```text
reward.py
```

Defines reinforcement-learning reward calculations.

---

## `src/data/`

Handles traffic data.

```text
collector.py
dataset.py
preprocessing.py
```

Important traffic features include:

```text
Vehicle Count
Waiting Time
Average Speed
Queue Length
```

---

## `src/prediction/`

Contains the LSTM implementation.

```text
lstm.py
predict.py
```

---

## `src/graph/`

Contains the GNN implementation.

```text
graph_builder.py
graph_dataset.py
gnn.py
```

The 16 intersections are represented as graph nodes, allowing the system to model spatial relationships between neighboring intersections.

---

## `src/agents/`

Contains MAPPO.

```text
actor.py
critic.py
buffer.py
mappo.py
```

The system uses multiple agents to control the traffic intersections.

---

## `src/emergency/`

Handles emergency and VIP traffic.

Main functionality includes:

* Vehicle detection
* Priority calculation
* Route handling
* Emergency scenario generation
* VIP scenario handling

---

## `src/evaluation/`

Contains result-processing and evaluation programs.

Important files:

```text
metrics.py
evaluate.py
comparison.py
final_results.py
final_emergency_comparison.py
plot_results.py
```

---

# 🖥️ Recommended Development Environment

### Recommended

**Visual Studio Code**

VS Code is recommended because it provides:

* Integrated terminal
* Python support
* Debugging
* Git/GitHub integration
* Easy project navigation

### Other IDEs

The project can also be opened using other Python-compatible IDEs, but the setup and commands in this README are written for **Windows + VS Code**.

---

#  Requirements

### Operating System

```text
Windows 10 / Windows 11
```

### Software

* Python 3.10+
* SUMO 1.27.1 recommended
* Git
* Visual Studio Code recommended

### Python packages

All Python dependencies are listed in:

```text
requirements.txt
```

---

# ⚙️ Installation

## 1. Clone the Repository

Open the VS Code terminal and run:

```powershell
git clone https://github.com/Harshul055/Adaptive-Traffic-Signal-Control-Simulation.git
```

Then enter the repository:

```powershell
cd Adaptive-Traffic-Signal-Control-Simulation
```

At this point, your terminal should be inside:

```text
<PROJECT_ROOT>
```

---

# 2. Open the Project in VS Code

From the project root:

```powershell
code .
```

If the `code` command is unavailable, open VS Code manually and select:

```text
File → Open Folder
```

Then select the project root.

---

# 3. Create a Python Virtual Environment

From the project root:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

After activation, the terminal should show something similar to:

```text
(venv)
```

---

# 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

---

# 5. Install Python Dependencies

From the project root:

```powershell
pip install -r requirements.txt
```

Wait until all packages finish installing.

---

# 6. Verify Python

```powershell
python --version
```

Example:

```text
Python 3.13.x
```

---

# 🚦 SUMO Installation and PATH Setup

SUMO can be installed in any suitable Windows directory.

The project does **not** require SUMO to be installed in a specific folder.

After installing SUMO, verify it from the VS Code terminal:

```powershell
sumo-gui --version
```

A working installation should return the installed SUMO version.

For example:

```text
Eclipse SUMO sumo-gui Version 1.27.1
```

---

## If `sumo-gui` is not recognized

Add the SUMO `bin` directory to the Windows PATH.

For example, if SUMO was installed here:

```text
C:\Program Files (x86)\Eclipse\Sumo
```

the required PATH entry is:

```text
C:\Program Files (x86)\Eclipse\Sumo\bin
```

The exact path depends on where SUMO was installed on your computer.

After updating PATH, restart VS Code and test:

```powershell
sumo-gui --version
```

---

# ▶️ Running the Project

## First: Go to the Project Root

Always make sure the VS Code terminal is inside the project root.

For your current setup:

```powershell
cd "C:\Project\Adaptive-Traffic-Control"
```

Then activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

---

# 🚀 Using `app.py`

`app.py` is the **main project launcher**.

Instead of remembering multiple individual commands, you can start the application using:

```powershell
python app.py
```

It provides a central menu for the major project operations.

The application can be used to access:

```text
SUMO Simulation
Traffic Data Collection
LSTM Training
GNN Training
MAPPO Training
Emergency/VIP Scenario
Scenario Merging
MAPPO Evaluation
Final Evaluation
Final Result Processing
Final Plot Generation
Complete Final Pipeline
Full Training + Evaluation Pipeline
Output Locations
```

### Start the application

From:

```text
<PROJECT_ROOT>
```

run:

```powershell
python app.py
```

---

#  Running SUMO Directly

To start SUMO-GUI manually:

```powershell
sumo-gui -c sumo/simulation/simulation.sumocfg
```

SUMO-GUI should display:

* Road network
* Vehicles
* Traffic signals
* Vehicle movement
* Signal phases

---

# 📊 Data Collection

Traffic data can be collected from SUMO using:

```powershell
python scripts\collect_data.py
```

The generated data is stored in:

```text
evaluation/
└── results/
    └── traffic_data.csv
```

Main collected features:

```text
vehicle_count
waiting_time
average_speed
queue_length
```

---

#  LSTM Training

The LSTM learns temporal traffic patterns.

Train the model:

```powershell
python training\train_lstm.py
```

The trained model is stored at:

```text
models/
└── lstm/
    └── traffic_lstm.pth
```

Generate predictions:

```powershell
python src\prediction\predict.py
```

Predictions are stored at:

```text
evaluation/
└── results/
    └── lstm_predictions.csv
```

---

#  GNN Training

The GNN represents the traffic network as a graph.

Each intersection is treated as a node and neighboring intersections are connected through graph relationships.

Train the GNN:

```powershell
python training\train_gnn.py
```

Model:

```text
models/
└── gnn/
    └── traffic_gnn.pth
```

The project uses 16 traffic-light intersections.

---

#  MAPPO Training

MAPPO treats each traffic-light intersection as an individual agent.

For this project:

```text
16 intersections
       ↓
16 MAPPO agents
```

Train MAPPO:

```powershell
python training\train_mappo.py
```

Models are stored in:

```text
models/
└── mappo/
    ├── actor_01.pth
    ├── ...
    ├── actor_16.pth
    ├── critic_01.pth
    └── ...
```

---

# 🚑 Emergency and VIP System

The project supports special vehicles including:

```text
Emergency
Ambulance
Fire
Police
VIP
```

Priority order:

```text
Emergency
     ↓
VIP
     ↓
Normal Traffic
```

Emergency traffic can receive priority over normal traffic-signal decisions.

---

# 🚨 Emergency/VIP Scenario

The tested scenario contains:

```text
5 Emergency Vehicles
        +
5 VIP Vehicles
        =
10 Special Vehicles
```

Generate the scenario:

```powershell
python scripts\generate_emergency_scenario.py
```

Merge the normal and special traffic:

```powershell
python scripts\merge_traffic_scenario.py
```

Test emergency detection:

```powershell
python scripts\test_emergency_detection.py
```

---

# 📈 Evaluation

The project compares:

```text
Fixed-Time Control
        VS
GNN + MAPPO + Emergency/VIP
```

The main metrics are:

| Metric        | Better Direction |
| ------------- | ---------------- |
| Vehicle Count | Lower            |
| Waiting Time  | Lower            |
| Average Speed | Higher           |
| Queue Length  | Lower            |

Run MAPPO evaluation:

```powershell
python scripts\run_mappo_evaluation.py
```

Run final evaluation:

```powershell
python scripts\evaluate.py
```

---

# 🏆 Final Results

The tested final scenario produced the following results:

| Metric        | Fixed-Time | GNN + MAPPO + Emergency/VIP |       Improvement |
| ------------- | ---------: | --------------------------: | ----------------: |
| Vehicle Count |      68.11 |                       32.12 |  **52.85% lower** |
| Waiting Time  |    2136.66 |                      108.95 |  **94.90% lower** |
| Average Speed |   6.72 m/s |                   13.14 m/s | **95.51% higher** |
| Queue Length  |      26.67 |                        8.78 |  **67.06% lower** |

These values represent the tested simulation scenario and are provided as the project's recorded evaluation results.

---

# 📁 Output Files

Important generated files include:

```text
evaluation/
│
├── results/
│   ├── traffic_data.csv
│   ├── lstm_predictions.csv
│   ├── fixed_time_results.csv
│   ├── mappo_results.csv
│   ├── mappo_final_results.csv
│   ├── comparison_results.csv
│   └── final_emergency_comparison.csv
│
├── plots/
│   └── final/
│
└── tables/
    └── final_results.csv
```

---

# 📊 Visualization

The project contains visualization scripts for analyzing the final results.

Generate final visualizations using:

```powershell
python src\final_visualizer.py
```

Additional evaluation plotting functionality is available through:

```powershell
python src\evaluation\plot_results.py
```

Final plots are stored under:

```text
evaluation\plots\
```

and final visualization output can be stored under:

```text
evaluation\plots\final\
```

---

# 🔄 Complete Execution Order

## From a Fresh Installation

Run the following steps from:

```text
<PROJECT_ROOT>
```

### Step 1 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 2 — Verify SUMO

```powershell
sumo-gui --version
```

### Step 3 — Test SUMO

```powershell
python scripts\run_sumo.py
```

### Step 4 — Collect traffic data

```powershell
python scripts\collect_data.py
```

### Step 5 — Train LSTM

```powershell
python training\train_lstm.py
```

### Step 6 — Generate LSTM predictions

```powershell
python src\prediction\predict.py
```

### Step 7 — Train GNN

```powershell
python training\train_gnn.py
```

### Step 8 — Train MAPPO

```powershell
python training\train_mappo.py
```

### Step 9 — Generate Emergency/VIP scenario

```powershell
python scripts\generate_emergency_scenario.py
```

### Step 10 — Merge traffic scenario

```powershell
python scripts\merge_traffic_scenario.py
```

### Step 11 — Run MAPPO evaluation

```powershell
python scripts\run_mappo_evaluation.py
```

### Step 12 — Run final evaluation

```powershell
python scripts\evaluate.py
```

### Step 13 — Process final results

```powershell
python src\evaluation\final_results.py
```

### Step 14 — Generate plots

```powershell
python src\evaluation\plot_results.py
```

---

# ⚡ Recommended Way When Models Already Exist

The repository already contains trained models.

Therefore, **you do not need to retrain the models just to reproduce an evaluation using those existing checkpoints**.

The simplified workflow is:

```text
Existing Models
      ↓
Generate Emergency/VIP Scenario
      ↓
Merge Traffic Scenario
      ↓
Run MAPPO Evaluation
      ↓
Final Evaluation
      ↓
Final Results
      ↓
Final Plots
```

You can also use:

```powershell
python app.py
```

and select the complete final-result pipeline.

---

# 📌 Main Files to Run for Final Results

| File                                     | Purpose                         |
| ---------------------------------------- | ------------------------------- |
| `app.py`                                 | Main project launcher           |
| `scripts\run_sumo.py`                    | Starts/tests SUMO               |
| `scripts\collect_data.py`                | Collects traffic data           |
| `training\train_lstm.py`                 | Trains LSTM                     |
| `training\train_gnn.py`                  | Trains GNN                      |
| `training\train_mappo.py`                | Trains MAPPO                    |
| `scripts\generate_emergency_scenario.py` | Generates emergency/VIP traffic |
| `scripts\merge_traffic_scenario.py`      | Combines traffic scenarios      |
| `scripts\run_mappo_evaluation.py`        | Evaluates trained MAPPO         |
| `scripts\evaluate.py`                    | Performs evaluation             |
| `src\evaluation\final_results.py`        | Processes final results         |
| `src\evaluation\plot_results.py`         | Generates final plots           |

---

# 💾 Existing Trained Models

The repository contains trained model files:

```text
models/
│
├── lstm/
│   └── traffic_lstm.pth
│
├── gnn/
│   └── traffic_gnn.pth
│
└── mappo/
    ├── actor_01.pth
    ├── ...
    ├── actor_16.pth
    ├── critic_01.pth
    └── ...
```

Therefore, a user can use the existing trained models for evaluation without starting the complete training process again.

---

# 🔧 Troubleshooting

## `No module named 'src'`

Make sure the terminal is inside the project root.

For your setup:

```powershell
cd "C:\Project\Adaptive-Traffic-Control"
```

Then run the command again.

For module-based Python tests, use:

```powershell
python -m src.agents.test_mappo
```

instead of directly executing the module file.

---

## SUMO not found

Check:

```powershell
sumo-gui --version
```

If Windows cannot find SUMO, add the SUMO `bin` directory to PATH.

Example:

```text
C:\Program Files (x86)\Eclipse\Sumo\bin
```

Restart VS Code after changing PATH.

---

## Python packages are missing

Run:

```powershell
python -m pip install --upgrade pip
```

Then:

```powershell
pip install -r requirements.txt
```

---

## Virtual environment is not activated

Run:

```powershell
.\venv\Scripts\Activate.ps1
```

Then verify:

```powershell
python --version
```

---

## `app.py` cannot find a file

Make sure `app.py` is located at the **project root**:

```text
Adaptive-Traffic-Control/
│
├── app.py
├── README.md
├── requirements.txt
├── src/
├── scripts/
├── training/
└── ...
```

Run it from the project root:

```powershell
python app.py
```

Do not move `app.py` into `src/`, `scripts/`, or `training/`.

---

# 📍 Important Path Rule

The project should always be treated relative to its root.

For example, if your project is:

```text
C:\Project\Adaptive-Traffic-Control
```

then:

```text
<PROJECT_ROOT>\app.py
```

means:

```text
C:\Project\Adaptive-Traffic-Control\app.py
```

and:

```text
<PROJECT_ROOT>\models\lstm\traffic_lstm.pth
```

means:

```text
C:\Project\Adaptive-Traffic-Control\models\lstm\traffic_lstm.pth
```

This allows the repository to be placed anywhere on a Windows computer without changing the project structure.

---

# 🎯 Project Summary

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
  +
Evaluation
```

to create an adaptive multi-intersection traffic signal control system.

The overall concept is:

```text
Traffic Simulation
       ↓
Traffic Data
       ↓
LSTM Prediction
       ↓
GNN Spatial Representation
       ↓
MAPPO Multi-Agent Control
       ↓
Traffic Signal Actions
       ↓
Emergency/VIP Priority
       ↓
SUMO Simulation
       ↓
Performance Evaluation
       ↓
Final Results
```

The project contains temporal prediction (LSTM), spatial traffic representation (GNN), and multi-agent reinforcement learning (MAPPO). The current matched paper evaluation uses the trained GNN + MAPPO controller with Emergency/VIP priority; the LSTM is not yet fused into the MAPPO policy input, so a GNN+LSTM+MAPPO performance claim requires a separate integration and retraining run.


---

# 📋 Matched Research-Paper Evaluation

For the final research-paper comparison, use:

```powershell
python scripts\run_paper_evaluation.py
```

This is different from the older MAPPO evaluation script. It performs two matched evaluations:

1. **Fixed-Time**
2. **GNN + MAPPO + Emergency/VIP Priority**

Both runs use the same generated traffic scenario and the same evaluation horizon.

The evaluator collects:

| Metric | Direction |
|---|---|
| Delay / Waiting Time | ↓ lower is better |
| Queue Length | ↓ lower is better |
| Average Speed | ↑ higher is better |
| Vehicle Count | ↓ lower in this experiment |
| CO₂ | ↓ lower is better |
| Throughput | ↑ higher is better |
| Emergency/VIP Priority Response | ↑ higher is better |

### CO₂

CO₂ is measured directly through SUMO/TraCI rather than estimated from average speed:

```text
CO2(g) =
Σ[CO2 emission rate (mg/s) × simulation step (s)] / 1000
```

### Throughput

```text
Throughput (veh/h) =
completed vehicles / simulation duration (s) × 3600
```

Completed vehicles are counted using SUMO's arrived-vehicle list.

### Emergency/VIP response

The evaluator records when an Emergency/VIP vehicle is detected and when priority is granted.

```text
Priority Response Rate =
priority grants / detected special vehicles × 100
```

It also reports average priority response time.

### Output

Raw matched evaluation data:

```text
evaluation/results/paper_fixed_time_metrics.csv
evaluation/results/paper_gnn_mappo_emergency_vip_metrics.csv
```

Final paper table:

```text
evaluation/tables/paper_results.csv
```

**Important:** The numerical CO₂ and throughput values must be obtained by running the SUMO evaluation. They should not be manually estimated from waiting time, queue length, or average speed.
