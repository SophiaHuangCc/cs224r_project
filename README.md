# CS224R Final Project: Reward Hacking in Embodied Agents

This repository contains the code for our Stanford CS224R final project:

> **Reward Hacking in Embodied Agents: Evaluating the Safety of LLM-Generated Reward Functions**

## Project Overview

Recent work such as Eureka shows that large language models (LLMs) can automatically generate reward functions for reinforcement learning tasks. However, LLMs are trained on text and may omit important physical constraints such as torque, force, and smooth motion limits.

In this project, we investigate:

1. Whether LLM-generated rewards lead to unsafe robot behaviors.
2. How often reward hacking occurs in robotic manipulation.
3. Whether simple mitigation strategies improve safety.

We evaluate these questions in simulation using robotic manipulation environments and PPO.

---

## Project Pipeline

```text
Task description
      ↓
GPT-4o generates reward function
      ↓
PPO trains policy in simulation
      ↓
Evaluate task success and safety metrics
      ↓
Apply mitigation strategies
      ↓
Compare results
```

---

## Repository Structure

```text
cs224r_project/
├── configs/                 # Hyperparameter and experiment configs
├── prompts/                 # Prompt templates for GPT reward generation
├── rewards/                 # Generated reward functions
├── scripts/                 # Training and evaluation scripts
├── logs/                    # TensorBoard and training logs
├── models/                  # Saved PPO checkpoints
├── results/                 # Plots, metrics, and tables
├── test_installation.py     # Verifies environment setup
├── train_ppo_baseline.py    # Train PPO using default environment reward
├── visualize_policy.py      # Render trained policy
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Environment Setup

### Create Conda Environment

```bash
conda create -n cs224r_project python=3.10 -y
conda activate cs224r_project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## requirements.txt

```txt
torch
numpy
matplotlib
pandas
tqdm
wandb

gymnasium
gymnasium-robotics
mujoco

stable-baselines3[extra]

openai
python-dotenv

mani_skill
mani_skill-nightly

modal
```

---

## Verify Installation

```bash
python test_installation.py
```

Expected output should show the observation and action spaces for `FetchReach-v4`.

---

## Running the Baseline

### Train PPO on FetchReach

```bash
python train_ppo_baseline.py
```

### Visualize the Trained Policy

```bash
python visualize_policy.py
```

---

## Alternative simulation: ManiSkill3

ManiSkill3 can run using CPU simulation and Vulkan-based rendering. Modal GPU simulation is also supported but no visualization available.

### Install ManiSkill

```bash
pip install mani_skill torch
```

### Test Local Rendering

```bash
python test_maniskill_local.py
```

### Test Modal Rendering

```bash
python test_maniskill_modal.py
```

---

## Initial Benchmark Tasks

1. Reach
2. Pick-and-Place
3. Lift-Fragile-Object (or equivalent custom task)

---

## Safety Metrics

During evaluation, we measure:

- Peak joint torque
- Contact force
- Joint velocity
- Jerk
- Workspace violations
- Self-collisions

---

## Baselines

1. LLM-generated reward (unconstrained)
2. LLM-generated reward with safety prompting
3. Human-engineered reward with explicit safety terms

---

## Mitigation Strategies

### Physics-Grounded Constraint Augmentation (PGCA)

Automatically append penalty terms for:
- torque violations
- excessive velocity
- large contact forces

### Constrained RL (CMDP)

Use PPO-Lagrangian to optimize task reward subject to safety constraints.

---

## Current Progress

- [x] Local Conda environment created
- [x] MuJoCo + Gymnasium-Robotics installed
- [x] Maniskill V3 installed
- [x] PPO baseline training on FetchReach
- [x] Visualization working
- [ ] Custom reward wrapper
- [ ] GPT-generated reward functions
- [ ] Safety metric logging
- [ ] Mitigation experiments

---

## Notes

This project uses **online reinforcement learning**, meaning no fixed dataset is required. The agent generates data by interacting with the simulator during training.

---

## References

- Eureka: Human-Level Reward Design via Coding Large Language Models
- Stable-Baselines3
- Gymnasium Robotics
- MuJoCo

---

## Authors

- Sophia Huang
- Project Partner
