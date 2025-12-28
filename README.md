# Federated Learning with DermaMNIST (Part 3)

This project implements **Conventional Federated Learning** with PID-based Client Detection using the **Flower (FLWR)** framework on the **DermaMNIST** dataset.
It extends the work done in Part 2 by introducing client anomaly detection and removal using a PID (Proportional–Integral–Derivative) control algorithm to defend against label-flipping and model-poisoning attacks by simulating multiple clients (both clean and poisoned) performing local training with **ResNet-18** models and aggregating results on a central server.

It fulfills the specifications outlined in *FL Project – Part 3*.

## Project Structure

```plaintext
Project_Part3/
│
├── Dataset_Derma/
│   ├── data_raw/                # Raw MedMNIST dataset (auto-downloaded)
│   ├── data_preprocessed/       # Generated clean and poisoned client datasets
│
├── Part3/
│   ├── src/
│   │   ├── derma.py             # Dataset Preprocessing and Model Initialization
│   │   ├── fl_client.py         # Flower client logic (training & evaluation)
│   │   ├── fl_server.py         # Server logic for both FedAvg and PID strategies
|   |   ├── server_pid.py        # PID-based client detection and removal strategy
│   │   ├── run_experiment.py    # Orchestrates clean & attack FL experiments
│   │   ├── plot_metrics.py      # Plots results and performance metrics
│   │   ├── initial_resnet18_pretrained.pth   # ResNet-18 model generated from Part1
|   |   ├── models/               # Saves all Client models generated during rounds.
│   ├── results/
│   │   ├── no_attack/           # Results from clean FL phase
│   │   │   ├── plots/           # All Plots related to no_attack are saved here
│   │   │   ├── per_clients_metrics.csv
│   │   │   ├── round_metrics.csv
│   │   ├── attack/              # Results from label-flipping attack phase
│   │   │   ├── plots/           # All Plots related to attack are saved here
│   │   │   ├── per_clients_metrics.csv
│   │   │   ├── round_metrics.csv
|   |   |   ├── removed_clients.csv # Clients removed via PID-based filtering
```

## Requirements

Install dependencies using the following command:

```bash
pip install torch torchvision flwr medmnist scikit-image pandas matplotlib seaborn imageio
```

## Step-by-Step Execution

### 1. Data Preparation

The data preparation is already completed in **Part 1**, and the preprocessed data from that part is used in this project. This process includes:

- Downloading the **DermaMNIST** dataset.
- Denoising, augmenting, and partitioning the dataset across 20 clients.
- Generates:
  - Clean client data (`dermaMNIST_clean/Client_i`)
  - Posioned data with label-flipping (`dermaMNIST_poison_flip/Client_i`)
- Saving metadata in `meta.json`.
- Generating the initial **ResNet-18** model (`initial_resnet18_pretrained.pth`).


### 2. Run Conventional Federated Learning Experiment

This script runs both the **No Attack** and **Attack** phases sequentially.
- Phase 1: Conventional Federated Learning (No Attack)
- Phase 2: Federated Learning with Label-Flipping Attack (using PID defense)

**Command:**

```bash
python run_experiment.py
```
**Execution Details**:
- Launch the Flower server (`fl_server.py`).
- Start a batch of 10 clients at a time from clean and poisoned datasets.
    - The poisoned clients [1, 2, 12, 15] (corrupted in Part 1) will always be included based on no.of clients being trained in Federated Learning, if only 10 clients are being trained then 4 of poisoned clients IDs will be included.
    - This is the flip map: {1: 0.1, 12: 0.25, 15: 0.5, 2: 0.75}.
    - The remaining 6 clients are selected randomly from the clean dataset.
- Run 3 communication rounds per phase.
- Save all metrics in the results/ directory under the following subdirectories:
    - `results/no_attack/`
    - `results/attack/`

### 3. PID-Based Client Detection (Part 3)
Implemented in (`server_pid.py`), the PID-based defense identifies anomalous or poisoned clients by monitoring deviations in their model updates and accuracies.
It detects and permanently removes poisoned clients based on their:
- L2 distance from the global model centroid.
- Historical deviation (PID controller).
- Accuracy deviation compared to other clients.

The **PID** controller then computes a score:

```math
PID = K_{p} \cdot d + K_{i} \cdot \sum d + K_{d} \cdot (d - d_{prev})
```

where:
- d: Normalized L2 distance from centroid.
- K_p, K_i, K_d :control coefficients

A client is removed **only if**:
- Its PID score exceeds the threshold (`pid_threshold`)
- Its accuracy falls significantly below (80%) of the group mean accuracy.


This ensures:
- Only poisoned clients with higher label-flip severity (e.g., 0.5 or 0.75 flipped labels) are detected and removed.
- Mildly poisoned clients (e.g., 0.1 or 0.25 flipped), whose behavior remains close to clean clients, are not being removed.
- Clean clients are retained even in noisy or heterogeneous conditions.

Removed clients are logged in:
```bash 
results/attack/removed_clients.csv
```

### 4. Plot and Analyze Results

After the experiment finishes, generate plots to visualize training and attack impact.
**Command:**

```bash
python plot_metrics.py
```

This will create:
- Per-client and round-level average accuracy/loss plots.
- Attack severity analysis (before vs after removal).
- These plots will be saved in:
     - `results/no_attack/plots/`
     - `results/attack/plots/`

### 4. Outputs:
The following files will be generated:

- **No Attack Phase**:
      - `results/no_attack/per_client_metrics.csv` — Metrics per client per round.
      - `results/no_attack/round_metrics.csv` — Aggregated round averages.

- **Attack Phase**:
     - `results/attack/per_client_metrics.csv` — Metrics per client per round.
     - `results/attack/round_metrics.csv` — Aggregated round averages.
     - `results/attack/removed_clients.csv` — List of removed clients with PID, distance, and accuracy info

- **Visualizations:**
- PNG plots for each phase saved in:
     - `results/no_attack/plots/`
     - `results/attack/plots/`











