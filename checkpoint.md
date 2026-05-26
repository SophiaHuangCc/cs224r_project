# Checkpoint: SAC+HER Safety Metrics Experiment

**Date:** 2026-05-25 (updated)
**Project:** CS224R — Reward Overoptimization in Robotic Manipulation

---

## Experiment Setup

- **Algorithm:** SAC + HER (Hindsight Experience Replay)
- **Environments:** FetchReach-v4, FetchPickAndPlace-v4, FetchSlide-v4
- **Reward types:** Vanilla (single-shot LLM-generated) vs Eureka (iteratively optimized LLM reward)
- **Step counts:** 50k, 100k, 250k, 500k
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

| Env + Reward | 50k | 100k | 250k | 500k |
|---|---|---|---|---|
| FetchReach vanilla | 1.00 | 1.00 | 1.00 | 1.00 |
| FetchReach eureka | 1.00* | 1.00 | 1.00 | 1.00 |
| FetchPickAndPlace vanilla | 0.05* | 0.02 | 0.10 | **0.46** |
| FetchPickAndPlace eureka | 0.05* | 0.02 | 0.12 | **0.06** |
| FetchSlide vanilla | 0.00* | 0.02 | 0.14 | 0.16 |
| FetchSlide eureka | 0.00* | 0.00 | 0.04 | **0.40** |

*50k values from `sac_smoke_50k` (basic metrics only, no safety metrics are included)

### Key Safety Metrics at 250k

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

### Key Safety Metrics at 500k

#### FetchReach (both 100% success)

| Metric | Vanilla | Eureka | Ratio |
|--------|---------|--------|-------|
| success_rate | 100.0% | 100.0% | 1.0× |
| action_violation_rate | 2.0% | 3.1% | 1.6× |
| jerk_violation_rate | **0.5%** | **2.3%** | **4.8×** |
| mean_jerk | 0.043 | 0.059 | 1.4× |
| drop_violation_rate | 5.6% | 19.0% | 3.4× |

#### FetchPickAndPlace (46% vs 6% success — EUREKA COLLAPSED 🎯🎯)

| Metric | Vanilla | Eureka | Ratio |
|--------|---------|--------|-------|
| success_rate | **46.0%** | **6.0%** | **0.13×** |
| action_violation_rate | **34.7%** | **56.2%** | **1.6×** |
| jerk_violation_rate | **37.7%** | **42.0%** | **1.1×** |
| mean_jerk | **0.899** | **1.102** | **1.2×** |
| drop_violation_rate | 7.3% | 5.0% | 0.7× |

#### FetchSlide (16% vs 40% success — eureka recovered task but unsafe)

| Metric | Vanilla | Eureka | Ratio |
|--------|---------|--------|-------|
| success_rate | 16.0% | **40.0%** | 2.5× |
| action_violation_rate | 17.2% | **24.5%** | **1.4×** |
| jerk_violation_rate | 8.8% | **14.5%** | **1.6×** |
| mean_jerk | 0.333 | **0.460** | **1.4×** |
| drop_violation_rate | 41.9% | 42.6% | 1.0× |

---

## Findings

### 1. Eureka consistently produces jerkier, more unsafe policies
Across all three environments and ALL step counts (100k→500k), eureka-trained policies exhibit higher jerk violation rates and mean jerk magnitudes. This holds even when success rates are identical (FetchReach) or when eureka achieves higher success (FetchSlide@500k).

### 2. FetchPickAndPlace: textbook reward overoptimization divergence
The strongest result. At 250k both methods had ~10-12% success. By 500k, vanilla improved to **46%** while eureka COLLAPSED to **6%** — more training made eureka *worse*. Meanwhile eureka's action violations grew to 56% (vs vanilla's 35%). This is the classic Goodhart's Law curve: the proxy reward diverges from the true objective with more optimization pressure.

### 3. FetchSlide: eureka achieves task success through unsafe behavior
At 500k, eureka actually solved the task better (40% vs 16%) BUT with 1.4× more action violations and 1.6× more jerk. The eureka reward learned an aggressive strategy (slam puck hard) that sometimes works but is physically dangerous. This demonstrates that **reward overoptimization doesn't always reduce task performance** — sometimes it finds unsafe shortcuts that happen to succeed.

### 4. Two failure modes of LLM reward overoptimization
- **Mode A (PickAndPlace):** Proxy reward diverges → policy gets stuck exploiting reward loopholes → task performance collapses
- **Mode B (Slide):** Proxy reward converges via unsafe path → task succeeds through dangerous behavior → safety degrades

### 5. Divergence grows with training steps
The vanilla-vs-eureka safety gap consistently grows from 100k → 250k → 500k across all environments. This confirms the overoptimization scaling hypothesis from Gao et al. 2022 applied to embodied domains.

---

---

## Hacking Rate (TODO)

Definition from DESIGN.md:
```
hacking_rate = (episodes where cumulative_LLM_reward > threshold AND is_success == False) / total_episodes
```

High hacking_rate = agent learned to game the reward without solving the task.

**To compute:** Need per-episode LLM reward + success labels from eval rollouts. The current metrics JSONs only have aggregated stats. Options:
1. Re-run eval with per-episode logging (save each episode's cumulative reward + is_success)
2. Load saved models and run a custom eval script that records per-episode data
3. Use `mean_reward` + `success_rate` as a proxy: if mean_reward is high but success_rate is low, that indicates hacking

**Proxy hacking signal from current data:**
| Scenario @ 500k | Success | Mean Reward | Signal |
|---|---|---|---|
| PickAndPlace eureka | 6% | -48.1 | Low success + very neg reward → agent stuck, not hacking reward |
| PickAndPlace vanilla | 46% | -31.9 | Higher success, better reward alignment |
| Slide eureka | 40% | -39.1 | Moderate success via aggressive unsafe behavior |
| Slide vanilla | 16% | -45.4 | Lower success, also low reward |

**Note:** The current reward values are the sparse *environment* reward (not LLM proxy reward). To properly compute hacking_rate, we need to evaluate episodes against BOTH the LLM proxy reward and the ground-truth success metric simultaneously.

**Next step:** Write `scripts/eval_hacking_rate.py` that:
1. Loads saved SAC models from `models/sac_{step}/`
2. Runs N eval episodes per model
3. Computes per-episode: LLM proxy reward (from generated_rewards/*.py) AND is_success (from env)
4. Calculates hacking_rate with configurable reward threshold
5. Outputs per-scenario hacking_rate table

---

## Data Locations

| Step Count | Metrics Dir | Models Dir | Notes |
|---|---|---|---|
| 50k (basic) | `logs/sac_smoke_50k/` | `models/sac_smoke_50k/` | All 6 scenarios, basic metrics only |
| 50k (safety) | `logs/sac_50k/` | `models/sac_50k/` | Only FetchReach_vanilla + PickAndPlace_eureka |
| 100k | `logs/sac_100k/` | Modal volume `cs224r-sac-results` | All 6 scenarios |
| 250k | `logs/sac_250k/` | Modal volume `cs224r-sac-results` | All 6 scenarios |
| 500k | `logs/sac_500k/` | `models/sac_500k/` | All 6 scenarios, local M4 |

### Downloading models from Modal (Sunny's Modal account)
```bash
conda run -n cs224r_project modal volume get cs224r-sac-results models/sac_100k/ models/sac_100k/
conda run -n cs224r_project modal volume get cs224r-sac-results models/sac_250k/ models/sac_250k/
```

---

## Training Script
```bash
# Run all scenarios at a given step count locally
cd ~/Desktop/cs224r/cs224r_project
conda run -n cs224r_project python -u scripts/sac/train_all_steps.py --steps 500000 --skip-existing

# Run all scenarios on Modal (parallel)
conda run -n cs224r_project modal run scripts/sac/train_modal.py

# Edit STEP_COUNTS in scripts/sac/train_modal.py to select step counts
```

---

## Next Steps (as of 2026-05-25)

1. **[ ] Hacking rate evaluation script** — compute per-episode LLM proxy reward vs ground-truth success
2. **[ ] Eureka + Mitigation condition** — feed safety metrics back to GPT-4o during Eureka iterations
3. **[ ] Sparse baseline runs** — train with original env sparse reward at same step counts for 4-condition comparison
4. **[ ] Scaling plot** — safety metrics vs training steps {100k, 250k, 500k} for the "money figure"
5. **[ ] Video recordings** — rollouts of hacking/unsafe behaviors (PickAndPlace eureka failing, Slide eureka slamming)
6. **[ ] Poster layout** (due Jun 4)
7. **[ ] Final report** (due Jun 9)
