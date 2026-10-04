# 🚦 Adaptive Traffic Signal Control Simulation

An AI-powered **Adaptive Traffic Signal Control System** built using **SUMO, GNN, LSTM, and MAPPO**, with dedicated **Emergency Vehicle and VIP Priority Management**.

The system simulates a road network containing **16 signalized intersections** and uses machine-learning and reinforcement-learning techniques to dynamically control traffic signals.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Project Objectives](#-project-objectives)
4. [Key Features](#-key-features)
5. [Technologies Used](#-technologies-used)
6. [System Architecture](#-system-architecture)
7. [How the Complete System Works](#-how-the-complete-system-works)
8. [Repository Structure](#-repository-structure)
9. [Prerequisites](#-prerequisites)
10. [SUMO Installation](#-sumo-installation)
11. [Python Environment Setup](#-python-environment-setup)
12. [Installing Dependencies](#-installing-dependencies)
13. [Project Configuration](#-project-configuration)
14. [Running SUMO](#-running-sumo)
15. [Traffic Data Collection](#-traffic-data-collection)
16. [LSTM Traffic Prediction](#-lstm-traffic-prediction)
17. [Graph Construction](#-graph-construction)
18. [GNN Processing](#-gnn-processing)
19. [MAPPO Traffic Control](#-mappo-traffic-control)
20. [Emergency Vehicle Priority](#-emergency-vehicle-priority)
21. [VIP Priority](#-vip-priority)
22. [Complete Scenario](#-complete-scenario)
23. [Evaluation](#-evaluation)
24. [Visualization](#-visualization)
25. [Final Results](#-final-results)
26. [Execution Order](#-recommended-execution-order)
27. [Important Files](#-important-files)
28. [Troubleshooting](#-troubleshooting)
29. [Common Commands](#-common-commands)
30. [Future Improvements](#-future-improvements)
31. [Conclusion](#-conclusion)

---

# 📖 Project Overview

Traditional traffic signals generally operate using:

* Fixed-time schedules
* Predefined signal phases
* Static timing
* Limited awareness of changing traffic conditions

These approaches cannot efficiently react to:

* Sudden traffic congestion
* Unequal traffic demand
* Long queues
* Slow-moving traffic
* Emergency vehicles
* VIP vehicles
* Traffic conditions at neighboring intersections

This project develops an **Adaptive Traffic Signal Control System** that observes traffic conditions and dynamically controls traffic signals.

The system combines:

```text
SUMO
  ↓
Traffic Simulation
  ↓
Data Collection
  ↓
LSTM Traffic Prediction
  ↓
Graph Construction
  ↓
GNN Spatial Feature Extraction
  ↓
MAPPO Multi-Agent Signal Control
  ↓
Emergency/VIP Priority
  ↓
Evaluation
  ↓
Visualization
```

---

# 🎯 Problem Statement

The objective is to reduce traffic congestion and improve traffic flow by dynamically controlling multiple traffic signals.

The system attempts to minimize:

* Vehicle waiting time
* Queue length
* Traffic congestion

while improving:

* Average vehicle speed
* Traffic throughput
* Emergency vehicle movement
* Coordination between neighboring intersections

---

# 🎯 Project Objectives

The main objectives are:

* Build a realistic multi-intersection SUMO environment.
* Collect real-time traffic information.
* Predict future traffic conditions using LSTM.
* Represent intersections as nodes in a graph.
* Extract spatial traffic relationships using GNN.
* Control multiple intersections using MAPPO.
* Detect emergency vehicles.
* Give emergency vehicles signal priority.
* Give VIP vehicles lower-priority preference.
* Compare AI-based control against fixed-time control.
* Generate quantitative results and visualizations.

---

# ⭐ Key Features

## 🚦 Multi-Intersection Control

The simulation contains:

```text
16 signalized intersections
```

organized as a 4 × 4 grid.

Each intersection can act as an independent MAPPO agent.

---

## 🧠 LSTM Traffic Prediction

LSTM is used to learn temporal traffic patterns.

The prediction system uses:

* Vehicle count
* Waiting time
* Average speed
* Queue length

to predict future traffic conditions.

---

## 🕸️ Graph Neural Network

Traffic intersections are represented as nodes.

Connections between intersections represent their spatial/traffic relationships.

The GNN learns information from:

* Current intersection
* Neighboring intersections
* Traffic conditions
* Network structure

---

## 🤖 MAPPO

MAPPO stands for:

**Multi-Agent Proximal Policy Optimization**

Each traffic signal is treated as an agent.

The agents cooperate to improve traffic flow across the complete network.

---

## 🚑 Emergency Vehicle Priority

The system detects:

* Ambulances
* Fire vehicles
* Police vehicles
* Emergency vehicles

When an emergency vehicle is detected, the system can override normal signal control and provide priority.

---

## ⭐ VIP Priority

VIP vehicles are also detected.

VIP priority is lower than emergency priority.

Priority hierarchy:

```text
Emergency
    ↓
VIP
    ↓
Normal Traffic
```

An emergency vehicle always receives priority over a VIP vehicle.

---

# 🛠️ Technologies Used

| Technology | Purpose                            |
| ---------- | ---------------------------------- |
| Python     | Main programming language          |
| SUMO       | Traffic simulation                 |
| TraCI      | Python ↔ SUMO communication        |
| PyTorch    | Deep learning                      |
| LSTM       | Traffic prediction                 |
| GNN        | Spatial traffic representation     |
| MAPPO      | Multi-agent reinforcement learning |
| NumPy      | Numerical processing               |
| Pandas     | Dataset processing                 |
| Matplotlib | Visualization                      |
| YAML       | Configuration                      |
| Git/GitHub | Version control                    |

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │        SUMO         │
                    │ Traffic Simulation  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Collector    │
                    │                     │
                    │ Vehicle Count       │
                    │ Waiting Time        │
                    │ Average Speed       │
                    │ Queue Length        │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
        ┌─────────────────┐       ┌─────────────────┐
        │      LSTM       │       │ Graph Builder   │
        │                 │       │                 │
        │ Temporal        │       │ Intersections  │
        │ Prediction      │       │ + Connections   │
        └────────┬────────┘       └────────┬────────┘
                 │                         │
                 │                         ▼
                 │                ┌─────────────────┐
                 │                │       GNN       │
                 │                │                 │
                 │                │ Spatial         │
                 │                │ Features        │
                 │                └────────┬────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │       MAPPO         │
                    │                     │
                    │ 16 Traffic Agents   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Signal Controller   │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
          🚑 Emergency                 ⭐ VIP
             Priority                  Priority
                  │                         │
                  └────────────┬────────────┘
                               ▼
                    ┌─────────────────────┐
                    │       SUMO          │
                    │ Updated Traffic     │
                    └─────────────────────┘
```

---

# 🔄 How the Complete System Works

The complete system works in multiple stages.

### Step 1 — SUMO Simulation

SUMO creates the road network, vehicles, traffic lights and vehicle movement.

### Step 2 — Data Collection

The system collects traffic information from SUMO.

### Step 3 — LSTM Prediction

Historical traffic information is processed by LSTM to understand temporal traffic patterns.

### Step 4 — Graph Construction

The 16 intersections are converted into graph nodes.

### Step 5 — GNN Processing

The GNN processes information from each intersection and its neighboring intersections.

### Step 6 — MAPPO Control

MAPPO agents determine signal actions.

### Step 7 — Emergency/VIP Handling

Emergency and VIP vehicles are detected.

Emergency vehicles receive the highest priority.

### Step 8 — Signal Update

The traffic signal phase is changed according to the control decision.

### Step 9 — Evaluation

The AI-controlled system is compared against fixed-time traffic control.

### Step 10 — Visualization

Graphs and result tables are generated.

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
│   ├── data/
│   ├── prediction/
│   ├── graph/
│   ├── agents/
│   ├── emergency/
│   ├── evaluation/
│   └── final_visualizer.py
│
├── training/
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

# 📂 Folder-by-Folder Explanation

## `configs/`

Contains project configuration files.

### `config.yaml`

General project configuration.

Used for storing common settings that should not be hard-coded throughout the project.

### `network.yaml`

Contains network-related configuration.

It can define:

* Network information
* Number of intersections
* Traffic network parameters

### `training.yaml`

Contains machine-learning/reinforcement-learning training settings.

---

# 🚦 `sumo/`

Contains the project's SUMO configuration.

## `sumo/network/`

Contains the road network.

### `nodes.nod.xml`

Defines the nodes/intersections.

### `edges.edg.xml`

Defines roads connecting nodes.

### `network.net.xml`

Generated SUMO network file.

This is the actual network loaded by SUMO.

---

## `sumo/routes/`

Contains vehicle routes.

### `traffic.rou.xml`

Normal traffic routes.

### `emergency.rou.xml`

Emergency vehicle route/type information.

---

## `sumo/simulation/`

### `simulation.sumocfg`

Main SUMO simulation configuration.

It tells SUMO:

* Which network to load
* Which routes to load
* Simulation settings
* Traffic-light configuration

---

## `sumo/tls/`

### `traffic_lights.add.xml`

Additional traffic-light configuration.

---

# 📦 `sumo-rl/`

Contains the SUMO-RL reference/network resources used by the project.

The project uses the RESCO `grid4x4` scenario as the multi-intersection traffic network.

Important location:

```text
sumo-rl/
└── sumo_rl/
    └── nets/
        └── RESCO/
            └── grid4x4/
```

The grid4x4 scenario provides a network containing 16 signalized intersections.

---

# 🧩 `src/environment/`

This folder contains the traffic-control environment.

## `sumo_env.py`

Main environment connecting Python with SUMO through TraCI.

Responsibilities include:

* Starting SUMO
* Connecting through TraCI
* Advancing simulation steps
* Reading traffic information
* Controlling traffic lights
* Managing environment state

---

## `state.py`

Creates and manages traffic-state information.

Typical traffic features include:

```text
Vehicle Count
Waiting Time
Average Speed
Queue Length
Traffic Signal State
Neighbor Information
```

---

## `action.py`

Defines traffic signal actions.

MAPPO selects an action and this module helps translate the action into traffic-signal behavior.

---

## `reward.py`

Defines the reinforcement-learning reward.

The reward encourages:

* Lower waiting time
* Lower queue length
* Better traffic flow
* Higher average speed

---

## `test_sumo_env.py`

Tests whether the SUMO environment works correctly.

---

# 📊 `src/data/`

Responsible for collecting and processing traffic data.

## `collector.py`

Collects traffic information from SUMO.

The collected information includes:

```text
vehicle_count
waiting_time
average_speed
queue_length
```

---

## `dataset.py`

Creates datasets suitable for machine-learning models.

---

## `preprocessing.py`

Prepares raw traffic data for machine learning.

Typical operations include:

* Cleaning data
* Handling missing values
* Normalization
* Feature preparation

---

# 🔮 `src/prediction/`

Contains traffic prediction code.

## `lstm.py`

Defines the LSTM neural network.

LSTM is responsible for learning **temporal patterns**.

For example:

```text
Traffic at t-9
Traffic at t-8
Traffic at t-7
...
Traffic at t-1
        ↓
      LSTM
        ↓
Future Traffic
```

---

## `predict.py`

Loads the trained LSTM model and generates predictions.

Output:

```text
evaluation/results/lstm_predictions.csv
```

---

# 🕸️ `src/graph/`

Contains graph-processing components.

## `graph_builder.py`

Builds the traffic network graph.

Each traffic intersection becomes a graph node.

For example:

```text
A0 ─── A1 ─── A2 ─── A3
│      │      │      │
B0 ─── B1 ─── B2 ─── B3
│      │      │      │
C0 ─── C1 ─── C2 ─── C3
│      │      │      │
D0 ─── D1 ─── D2 ─── D3
```

---

## `graph_dataset.py`

Prepares graph data for GNN processing.

---

## `gnn.py`

Defines the Graph Neural Network.

The GNN learns spatial relationships between intersections.

Instead of looking at an intersection independently, it can incorporate information from neighboring intersections.

---

# 🤖 `src/agents/`

Contains the MAPPO implementation.

## `actor.py`

The Actor decides which action should be performed.

Conceptually:

```text
Traffic State
     ↓
   Actor
     ↓
Signal Action
```

---

## `critic.py`

The Critic evaluates the quality of the current state/action situation.

Conceptually:

```text
Traffic State
     ↓
   Critic
     ↓
State Value
```

---

## `buffer.py`

Stores reinforcement-learning experience.

The buffer can contain information such as:

* States
* Actions
* Rewards
* Log probabilities
* Values

---

## `mappo.py`

Contains the MAPPO agent implementation.

MAPPO allows multiple traffic-light agents to learn simultaneously.

---

# 🚑 `src/emergency/`

Handles emergency and VIP vehicles.

## `detector.py`

Detects special vehicles.

Supported emergency types include:

```text
emergency
ambulance
fire
police
```

VIP vehicles are recognized as:

```text
vip
```

---

## `priority.py`

Controls traffic-signal priority.

Priority order:

```text
Emergency > VIP > Normal
```

Emergency priority can override normal MAPPO signal decisions when necessary.

---

## `route.py`

Handles emergency route information.

---

## `route_generator.py`

Generates special vehicle routes.

---

## `scenario_generator.py`

Generates emergency/VIP traffic scenarios.

---

## `scenario_routes.py`

Handles routes used by emergency/VIP scenarios.

---

# 📈 `src/evaluation/`

Contains evaluation and result-generation modules.

## `metrics.py`

Calculates traffic-performance metrics.

---

## `evaluate.py`

Evaluates the trained system.

---

## `comparison.py`

Compares different traffic-control approaches.

---

## `final_results.py`

Generates final result information.

---

## `final_emergency_comparison.py`

Compares:

```text
Fixed-Time Control
        vs
GNN + MAPPO + Emergency/VIP
```

---

## Plot scripts

The following files generate different visualizations:

```text
plot_control.py
plot_lstm.py
plot_results.py
```

---

# 🖥️ `src/final_visualizer.py`

This is the main visualization generator.

It can generate the final project plots automatically from the evaluation data.

Generated plots are stored inside:

```text
evaluation/plots/final/
```

---

# 🏋️ `training/`

Contains model-training scripts.

## `train_lstm.py`

Trains the LSTM traffic prediction model.

Output:

```text
models/lstm/traffic_lstm.pth
```

---

## `train_gnn.py`

Trains the Graph Neural Network.

Output:

```text
models/gnn/traffic_gnn.pth
```

---

## `train_mappo.py`

Trains the MAPPO traffic-control agents.

The project uses 16 agents corresponding to the 16 traffic intersections.

Outputs include:

```text
models/mappo/actor_01.pth
...
models/mappo/actor_16.pth

models/mappo/critic_01.pth
...
models/mappo/critic_16.pth
```

---

# 💾 `models/`

Contains trained machine-learning models.

```text
models/
├── lstm/
│   └── traffic_lstm.pth
│
├── gnn/
│   └── traffic_gnn.pth
│
└── mappo/
    ├── actor_01.pth
    ├── actor_02.pth
    ├── ...
    ├── actor_16.pth
    ├── critic_01.pth
    ├── critic_02.pth
    ├── ...
    └── critic_16.pth
```

The models are already included in this repository.

Therefore, retraining is **not required just to run/evaluate the existing trained system**, provided the required environment and dependencies are configured.

---

# 📊 `evaluation/`

Contains generated results.

## `evaluation/results/`

Important files:

```text
traffic_data.csv
lstm_predictions.csv
fixed_time_results.csv
mappo_results.csv
mappo_final_results.csv
comparison_results.csv
final_emergency_comparison.csv
```

---

## `evaluation/plots/`

Contains generated graphs.

The `final/` directory contains the final visualization set.

---

## `evaluation/tables/`

Contains final result tables.

---

# 📜 `scripts/`

Contains convenient scripts for running different parts of the project.

## `run_sumo.py`

Runs the SUMO simulation.

---

## `collect_data.py`

Collects traffic data from SUMO.

---

## `train.py`

Provides a training entry point.

---

## `evaluate.py`

Runs evaluation.

---

## `generate_emergency_scenario.py`

Creates emergency scenarios.

---

## `merge_traffic_scenario.py`

Combines the normal traffic scenario with special emergency/VIP vehicles.

The final scenario contains:

```text
5 Emergency Vehicles
5 VIP Vehicles
----------------
10 Special Vehicles
```

---

## `run_mappo_evaluation.py`

Runs MAPPO evaluation.

---

## `run_scenario.py`

Runs a selected traffic scenario.

---

## `test_emergency_detection.py`

Tests emergency/VIP detection.

---

# 💻 Prerequisites

Before running the project, install:

### Required

* Windows 10/11
* Python 3.10+ recommended
* SUMO
* Git
* VS Code recommended

Recommended:

```text
Python 3.13
SUMO 1.27.1
```

---

# 🚦 SUMO Installation

Download and install SUMO from the official Eclipse SUMO project.

After installation, verify:

```powershell
sumo --version
```

or:

```powershell
sumo-gui --version
```

Expected output should show the installed SUMO version.

If `sumo` is not recognized, add the SUMO `bin` directory to the Windows PATH.

Typical installation:

```text
C:\Program Files (x86)\Eclipse\Sumo\bin
```

---

# 🐍 Python Setup

Clone the repository:

```powershell
git clone https://github.com/Harshul055/Adaptive-Traffic-Signal-Control-Simulation.git
```

Enter the project:

```powershell
cd Adaptive-Traffic-Signal-Control-Simulation
```

Check Python:

```powershell
python --version
```

---

# 🔐 Create Virtual Environment

Recommended:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

---

# 📦 Install Dependencies

Run:

```powershell
pip install -r requirements.txt
```

Verify important packages:

```powershell
pip list
```

---

# ⚙️ Project Configuration

Before running training or simulation, verify:

```text
configs/config.yaml
configs/network.yaml
configs/training.yaml
```

Also verify the SUMO executable path if it is hard-coded in any script.

Typical path:

```text
C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe
```

---

# ▶️ Running SUMO

For the standard project SUMO configuration, use:

```powershell
sumo-gui -c sumo/simulation/simulation.sumocfg
```

This opens the SUMO graphical interface.

You should see:

* Roads
* Intersections
* Traffic lights
* Vehicles
* Vehicle movement

---

# 📊 Collecting Traffic Data

Run:

```powershell
python scripts/collect_data.py
```

The collector communicates with SUMO through TraCI and records traffic information.

The output is:

```text
evaluation/results/traffic_data.csv
```

Important columns include:

```text
vehicle_count
waiting_time
average_speed
queue_length
```

---

# 🔮 Training LSTM

The LSTM learns temporal traffic behavior.

Run:

```powershell
python training/train_lstm.py
```

The trained model is saved as:

```text
models/lstm/traffic_lstm.pth
```

---

# 🔍 Running LSTM Prediction

After training or using the included trained model:

```powershell
python src/prediction/predict.py
```

Predictions are saved to:

```text
evaluation/results/lstm_predictions.csv
```

---

# 🕸️ Building the Traffic Graph

The traffic network contains 16 intersections.

The graph represents:

```text
Node = Intersection
Edge = Relationship between intersections
```

The graph builder identifies the traffic-light-controlled intersections and their relationships.

---

# 🧠 Training GNN

Run:

```powershell
python training/train_gnn.py
```

The trained model is saved as:

```text
models/gnn/traffic_gnn.pth
```

The GNN produces learned spatial representations for the intersections.

---

# 🤖 MAPPO Traffic Control

MAPPO treats each intersection as an independent agent.

For 16 intersections:

```text
16 Intersections
       ↓
16 MAPPO Agents
```

Each agent observes traffic conditions and selects a traffic-signal action.

Run training:

```powershell
python training/train_mappo.py
```

The resulting models are stored in:

```text
models/mappo/
```

---

# 🎛️ MAPPO Action Flow

Conceptually:

```text
Traffic State
      ↓
LSTM Prediction
      ↓
GNN Spatial Features
      ↓
MAPPO Agent
      ↓
Action
      ↓
Traffic Signal
      ↓
SUMO
      ↓
New Traffic State
```

The process repeats continuously.

---

# 🚑 Emergency Vehicle Priority

The system can identify emergency vehicles.

Supported types:

```text
emergency
ambulance
fire
police
```

When an emergency vehicle approaches an intersection:

```text
Emergency detected
       ↓
Find relevant traffic signal
       ↓
Check current phase
       ↓
Determine green phase
       ↓
Give priority
       ↓
Emergency vehicle passes
```

This mechanism can override normal traffic-control behavior.

---

# ⭐ VIP Priority

VIP vehicles are handled separately.

The priority hierarchy is:

```text
                 ┌───────────────┐
                 │   Emergency   │
                 └───────┬───────┘
                         │ Highest
                         ▼
                 ┌───────────────┐
                 │      VIP      │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Normal Traffic│
                 └───────────────┘
```

VIP priority is provided only when emergency traffic does not require priority.

---

# 🚨 Complete Emergency/VIP Scenario

The final special-vehicle scenario contains:

```text
5 Emergency Vehicles
+
5 VIP Vehicles
=
10 Special Vehicles
```

Their departures are distributed during the early part of the simulation.

The scenario is generated using:

```powershell
python scripts/merge_traffic_scenario.py
```

The script creates a combined traffic scenario containing:

* Normal vehicles
* Emergency vehicles
* VIP vehicles

---

# 🧪 Testing Emergency Detection

Run:

```powershell
python scripts/test_emergency_detection.py
```

This checks whether special vehicles are correctly detected.

---

# 🧪 Testing the SUMO Environment

Run:

```powershell
python src/environment/test_sumo_env.py
```

If Python package imports cause an error, run module-based tests from the project root where applicable:

```powershell
python -m src.environment.test_sumo_env
```

---

# 📈 Evaluation

The project compares:

```text
Fixed-Time Traffic Control
              VS
GNN + MAPPO + Emergency/VIP
```

The primary metrics are:

### Vehicle Count

Measures the number of vehicles present in the simulation.

### Waiting Time

Measures accumulated vehicle waiting time.

Lower is better.

### Average Speed

Measures average vehicle speed.

Higher is better.

### Queue Length

Measures traffic queue size.

Lower is better.

---

# 📊 Final Results

The integrated final scenario produced the following average results:

| Metric        | Fixed-Time | GNN + MAPPO + Emergency/VIP |       Improvement |
| ------------- | ---------: | --------------------------: | ----------------: |
| Vehicle Count |      68.11 |                       32.12 |  **52.85% lower** |
| Waiting Time  |    2136.66 |                      108.95 |  **94.90% lower** |
| Average Speed |   6.72 m/s |                   13.14 m/s | **95.51% higher** |
| Queue Length  |      26.67 |                        8.78 |  **67.06% lower** |

### Interpretation

The AI-based system significantly reduces congestion-related metrics.

The largest improvements are observed in:

* Waiting time
* Average speed
* Queue length

while also reducing the average number of vehicles present in the network.

---

# 📉 Generated Visualizations

The project generates multiple visualization categories.

Examples include:

```text
01_vehicle_count.png
02_waiting_time.png
03_average_speed.png
04_queue_length.png
05_overall_improvement.png
06_horizontal_improvement.png
07_normalized_comparison.png
08_vehicle_count_over_time.png
09_waiting_time_over_time.png
10_average_speed_over_time.png
11_queue_length_over_time.png
12_all_traffic_metrics.png
13_vehicle_distribution.png
14_waiting_distribution.png
15_speed_distribution.png
16_queue_distribution.png
17_correlation_heatmap.png
19_final_results_table.png
20_final_project_gains.png
```

They are stored in:

```text
evaluation/plots/final/
```

---

# 🖼️ Generating Final Visualizations

Run:

```powershell
python src/final_visualizer.py
```

The script reads the generated CSV files and produces the final plots automatically.

---

# 🔁 Recommended Complete Execution Order

If setting up the project from scratch, follow this order.

## Step 1 — Clone

```powershell
git clone https://github.com/Harshul055/Adaptive-Traffic-Signal-Control-Simulation.git
cd Adaptive-Traffic-Signal-Control-Simulation
```

## Step 2 — Create environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## Step 3 — Install packages

```powershell
pip install -r requirements.txt
```

## Step 4 — Verify SUMO

```powershell
sumo-gui --version
```

## Step 5 — Test SUMO

```powershell
sumo-gui -c sumo/simulation/simulation.sumocfg
```

## Step 6 — Collect traffic data

```powershell
python scripts/collect_data.py
```

## Step 7 — Train LSTM

```powershell
python training/train_lstm.py
```

## Step 8 — Generate predictions

```powershell
python src/prediction/predict.py
```

## Step 9 — Train GNN

```powershell
python training/train_gnn.py
```

## Step 10 — Generate emergency/VIP scenario

```powershell
python scripts/merge_traffic_scenario.py
```

## Step 11 — Test emergency detection

```powershell
python scripts/test_emergency_detection.py
```

## Step 12 — Train MAPPO

```powershell
python training/train_mappo.py
```

## Step 13 — Run MAPPO evaluation

```powershell
python scripts/run_mappo_evaluation.py
```

## Step 14 — Generate evaluation results

```powershell
python scripts/evaluate.py
```

## Step 15 — Generate final visualizations

```powershell
python src/final_visualizer.py
```

---

# ⚡ Quick Start Using Existing Trained Models

The repository already contains trained models.

Therefore, if the goal is to inspect or reproduce the existing project rather than retrain everything, you can use the included:

```text
models/lstm/traffic_lstm.pth
models/gnn/traffic_gnn.pth
models/mappo/*.pth
```

This avoids unnecessarily retraining the models.

---

# 🔧 Important Difference: Training vs Evaluation

## Training

Training changes the model parameters.

Examples:

```powershell
python training/train_lstm.py
python training/train_gnn.py
python training/train_mappo.py
```

---

## Evaluation

Evaluation uses trained models to measure performance.

Examples:

```powershell
python scripts/run_mappo_evaluation.py
python scripts/evaluate.py
```

---

# 🧠 What Each AI Component Actually Does

## LSTM

### Main responsibility:

**Temporal prediction**

It answers:

> "Based on previous traffic conditions, what is the likely future traffic condition?"

---

## GNN

### Main responsibility:

**Spatial relationship learning**

It answers:

> "How does traffic at neighboring intersections affect this intersection?"

---

## MAPPO

### Main responsibility:

**Decision making**

It answers:

> "Given the current traffic state, what signal-control action should this intersection take?"

---

## Emergency Priority

### Main responsibility:

**Safety-critical override**

It answers:

> "Is an emergency vehicle approaching, and should the normal signal decision be overridden?"

---

## VIP Priority

### Main responsibility:

**Secondary priority management**

It provides controlled priority for VIP vehicles while remaining below emergency priority.

---

# 🔄 Complete AI Decision Pipeline

```text
              Historical Traffic
                     │
                     ▼
                  ┌─────┐
                  │ LSTM│
                  └──┬──┘
                     │
              Future Traffic
                     │
                     ▼
Traffic Network → ┌─────┐
                  │ GNN │
                  └──┬──┘
                     │
             Spatial Features
                     │
                     ▼
                 ┌───────┐
                 │ MAPPO │
                 └───┬───┘
                     │
              Signal Action
                     │
                     ▼
          Emergency/VIP Check
                     │
             ┌───────┴───────┐
             │               │
        Emergency          No Emergency
             │               │
             ▼               ▼
       Priority       Normal/VIP Control
             │               │
             └───────┬───────┘
                     ▼
                    SUMO
                     │
                     ▼
              Updated Traffic
                     │
                     └───────► Repeat
```

---

# 🧪 Troubleshooting

## `No module named 'src'`

Run commands from the project root:

```powershell
cd C:\Project\Adaptive-Traffic-Control
```

For module-based Python scripts, use:

```powershell
python -m src.agents.test_mappo
```

instead of directly executing a file when package imports require the project root.

---

# ❌ `sumo` is not recognized

Check:

```powershell
sumo --version
```

If it fails, add:

```text
C:\Program Files (x86)\Eclipse\Sumo\bin
```

to the Windows PATH.

Restart VS Code/PowerShell after changing PATH.

---

# ❌ SUMO/TraCI connection error

Check:

1. SUMO is installed.
2. SUMO executable path is correct.
3. The SUMO configuration file exists.
4. The network file exists.
5. The route file exists.
6. Another SUMO instance is not occupying the connection.

---

# ❌ SUMO GUI closes immediately

Run SUMO directly:

```powershell
sumo-gui -c sumo/simulation/simulation.sumocfg
```

Check the terminal for errors.

---

# ❌ Model file not found

Verify:

```text
models/
├── lstm/
├── gnn/
└── mappo/
```

The repository already contains trained model files.

---

# ❌ Python package installation error

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Then:

```powershell
pip install -r requirements.txt
```

---

# ❌ Git shows unwanted Python cache files

Python cache files should not be committed.

Remove them:

```powershell
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force
```

The `.gitignore` file prevents future cache files from being committed.

---

# 📁 Important Output Locations

## Traffic dataset

```text
evaluation/results/traffic_data.csv
```

## LSTM predictions

```text
evaluation/results/lstm_predictions.csv
```

## Fixed-time results

```text
evaluation/results/fixed_time_results.csv
```

## MAPPO results

```text
evaluation/results/mappo_final_results.csv
```

## Emergency comparison

```text
evaluation/results/final_emergency_comparison.csv
```

## Final plots

```text
evaluation/plots/final/
```

## LSTM model

```text
models/lstm/traffic_lstm.pth
```

## GNN model

```text
models/gnn/traffic_gnn.pth
```

## MAPPO models

```text
models/mappo/
```

---

# 🧮 Main Evaluation Metrics

The project focuses on four primary traffic metrics:

### 1. Vehicle Count

Indicates the number of vehicles present in the simulation.

Lower average congestion-related vehicle count is generally preferred in the evaluated scenario.

### 2. Waiting Time

Indicates how much time vehicles spend waiting.

```text
Lower = Better
```

### 3. Average Speed

Indicates how efficiently vehicles move through the network.

```text
Higher = Better
```

### 4. Queue Length

Indicates congestion at intersections.

```text
Lower = Better
```

---

# 🏆 Project Outcome

The final integrated system combines:

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

The resulting system provides adaptive multi-intersection traffic control instead of relying only on fixed-time signals.

The final evaluation demonstrates substantial improvements over the fixed-time baseline in the tested scenario.

---

# 🚀 Future Improvements

Possible future improvements include:

* Real-world traffic-camera integration
* Real-time vehicle detection
* YOLO-based traffic detection
* GPS-based emergency vehicle detection
* Real-world traffic signal controller integration
* Larger road networks
* More intersections
* Multi-city traffic datasets
* Transformer-based traffic prediction
* Advanced multi-objective reinforcement learning
* Cloud-based monitoring
* Live traffic dashboard
* Edge/IoT deployment
* Real-time adaptive routing
* Weather-aware traffic control
* Accident detection
* Pedestrian-aware signal control

---

# 🔐 Reproducibility

For reproducible experiments:

1. Use the same SUMO network.
2. Use the same route files.
3. Use the same configuration files.
4. Use the same random seeds where provided.
5. Use the included trained models when reproducing the reported evaluation.
6. Keep the same metric calculation scripts.
7. Generate visualizations from the resulting CSV files.

---

# 📌 Important Note

The reported performance values are based on the project's tested simulation scenarios.

They should not automatically be interpreted as guaranteed real-world traffic improvements.

A real-world deployment would require:

* Real traffic data
* Traffic-signal hardware integration
* Safety validation
* Regulatory approval
* Real-world testing
* Fail-safe signal control
* Extensive simulation under many traffic conditions

---

# 👨‍💻 Project

**Adaptive Traffic Signal Control Simulation**

Core concept:

> Use temporal traffic prediction, spatial traffic intelligence, multi-agent reinforcement learning, and priority-based emergency handling to create an adaptive traffic signal control system.

---



---

# ⭐ Summary

```text
                ADAPTIVE TRAFFIC CONTROL
                         │
          ┌──────────────┼──────────────┐
          │              │              │
         LSTM           GNN           MAPPO
          │              │              │
      Prediction      Spatial       Decision
          │           Learning        Making
          │              │              │
          └──────────────┼──────────────┘
                         │
                  Traffic Signals
                         │
                ┌────────┴────────┐
                │                 │
            Emergency            VIP
             Priority          Priority
                │                 │
                └────────┬────────┘
                         │
                        SUMO
                         │
                    Evaluation
                         │
                  Final Results
```

**The repository contains the complete simulation, machine-learning models, reinforcement-learning agents, emergency/VIP handling, evaluation pipeline, results, and visualization system.**
