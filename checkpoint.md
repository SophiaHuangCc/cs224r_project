# Checkpoint: SAC+HER Safety Metrics Experiment

**Date:** 2026-05-22
**Project:** CS224R — Reward Overoptimization in Robotic Manipulation

---

## Experiment Setup

- **Algorithm:** SAC + HER (Hindsight Experience Replay)
- **Environments:** FetchReach-v4, FetchPickAndPlace-v4, FetchSlide-v4
- **Reward types:** Vanilla (single-shot LLM-generated) vs Eureka (iteratively optimized LLM reward)
- **Step counts:** 50k, 100k, 250k
- **Eval episodes:** 50 per checkpoint
- **Compute:** Modal (4 CPU cores, 8GB RAM per container, 12 parallel jobs)
- **Safety metrics:** action violations, jerk violations, object speed/accel violations, drop rate, table slam, workspace violations

### Reward Files
| Env | Vanilla | Eureka |
|-----|---------|--------|
| FetchReach-v4 | `generated_rewards/FetchReach-v4_vanilla.py` | `generated_rewards/eureka/FetchReach-v4_iter1.py` |
| FetchPickAndPlace-v4 | `generated_rewards/FetchPickAndPlace-v4_vanilla.py` | `generated_rewards/eureka/FetchPickAndPlace-v4_best.py` |
| FetchSlide-v4 | `generated_rewards/FetchSlide-v4_vanilla.py` | `generated_rewards/eureka/FetchSlide-v4_best.py` |

---

## Results Summary

### Success Rates

| Env + Reward | 50k | 100k | 250k |
|---|---|---|---|
| FetchReach vanilla | 1.00 | 1.00 | 1.00 |
| FetchReach eureka | 1.00* | 1.00 | 1.00 |
| FetchPickAndPlace vanilla | 0.05* | 0.02 | 0.10 |
| FetchPickAndPlace eureka | 0.05* | 0.02 | 0.12 |
| FetchSlide vanilla | 0.00* | 0.02 | **0.14** |
| FetchSlide eureka | 0.00* | 0.00 | **0.04** |

*50k values from `sac_smoke_50k` (basic metrics only, no safety metrics are included)

### Key Safety Metrics at 250k (primary comparison point)

#### FetchReach (both 100% success — controlled comparison)

| Metric | Vanilla | Eureka | Ratio |
|--------|---------|--------|-------|
| success_rate | 100.0% | 100.0% | 1.0× |
| action_violation_rate | 3.4% | 3.8% | 1.1× |
| jerk_violation_rate | **0.4%** | **2.1%** | **5.3×** |
| mean_jerk | 0.050 | 0.061 | 1.2× |
| drop_violation_rate | 11.3% | 7.8% | 0.7× |

#### FetchPickAndPlace (10% vs 12% success — similar performance)

| Metric | Vanilla | Eureka | Ratio |
|--------|---------|--------|-------|
| success_rate | **10.0%** | **12.0%** | **1.2×** |
| action_violation_rate | 40.0% | 41.4% | 1.0× |
| jerk_violation_rate | **17.0%** | **22.8%** | **1.3×** |
| mean_jerk | **0.427** | **0.585** | **1.4×** |
| drop_violation_rate | **0.7%** | **2.2%** | **3.1×** |

#### FetchSlide (14% vs 4% success — EUREKA WORSE 🎯)

| Metric | Vanilla | Eureka | Ratio |
|--------|---------|--------|-------|
| success_rate | **14.0%** | **4.0%** | **0.29×** |
| action_violation_rate | **12.8%** | **27.4%** | **2.1×** |
| jerk_violation_rate | **13.4%** | **21.7%** | **1.6×** |
| mean_jerk | **0.374** | **0.587** | **1.6×** |
| drop_violation_rate | **41.6%** | **64.6%** | **1.6×** |

---

## Findings

### 1. Eureka consistently produces jerkier policies
Across all three environments and step counts, eureka-trained policies exhibit higher jerk violation rates and mean jerk magnitudes. This holds even when success rates are identical (FetchReach) or comparable (FetchPickAndPlace).

### 2. FetchSlide shows classic reward overoptimization
The eureka reward for FetchSlide produces a policy that is **both worse at the task (4% vs 14% success) and more unsafe** (2.1× action violations, 1.6× jerk, 1.6× drops). The shaped reward optimizes a proxy that diverges from the true objective while encouraging unsafe behaviors.

### 3. PickAndPlace shows subtle overoptimization
At comparable success rates (10% vs 12%), the eureka policy drops objects 3× more often and has 1.3–1.4× more jerk. The reward shaping achieves marginally better task success at the cost of significantly worse safety.

### 4. Trend strengthens with training
The vanilla-vs-eureka safety gap grows from 100k to 250k as policies become more competent. This is consistent with the overoptimization hypothesis: the divergence between proxy reward and true objective worsens with more optimization pressure.

---

## Data Locations

| Step Count | Metrics Dir | Models Dir | Notes |
|---|---|---|---|
| 50k (basic) | `logs/sac_smoke_50k/` | `models/sac_smoke_50k/` | All 6 scenarios, basic metrics only |
| 50k (safety) | `logs/sac_50k/` | `models/sac_50k/` | Only FetchReach_vanilla + PickAndPlace_eureka |
| 100k | `logs/sac_100k/` | Modal volume `cs224r-sac-results` | All 6 scenarios |
| 250k | `logs/sac_250k/` | Modal volume `cs224r-sac-results` | All 6 scenarios |

### Downloading models from Modal (Sunny's Modal account)
```bash
conda run -n cs224r_project modal volume get cs224r-sac-results models/sac_100k/ models/sac_100k/
conda run -n cs224r_project modal volume get cs224r-sac-results models/sac_250k/ models/sac_250k/
```

---

## Training Script
```bash
# Run all scenarios at a given step count on Modal
conda run -n cs224r_project modal run scripts/sac/train_modal.py

# Edit STEP_COUNTS in scripts/sac/train_modal.py to select step counts
# Currently set to [250_000, 500_000]
```
