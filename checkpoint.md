# Checkpoint: Reward Overoptimization in Robotic Manipulation

**Date:** 2026-05-30
**Project:** CS224R Final Project — Sunny Yuan & Sophia Huang

---

## Methods

| # | Method | How reward is generated | How policy is trained | Notes |
|---|--------|------------------------|----------------------|-------|
| 1 | **Vanilla LLM** | GPT-4o single-shot, one reward function | SAC+HER | Baseline LLM reward |
| 2 | **Eureka (PPO)** | GPT-4o iterative (3 iters × 50k PPO eval) | SAC+HER | Reward from PPO loop, but trained with SAC+HER. Buggy HER wrapper. |
| 3 | **Eureka (SAC+HER)** | GPT-4o iterative (3 iters × 50k SAC+HER eval) | SAC+HER | Fixed HER-aware wrapper. Best single-reward method. |
| 4 | **Ensemble (min, N=3)** | 3× independent Eureka SAC+HER loops → min-aggregate | SAC+HER | Mitigation attempt. |

---

## Results: FetchReach-v4 (Easy)

| Method | 100k | 250k | 500k | Hacking | Action Viol | Jerk Viol |
|--------|------|------|------|---------|-------------|-----------|
| Vanilla LLM | 100% | 100% | 100% | 0% | 2-3% | 0.5% |
| Eureka (PPO reward) | 100% | 100% | 100% | 0% | 3-4% | 1.5-2.3% |
| Eureka (SAC+HER) | 100% | 100% | 100% | 0% | 3.3% | 1.7-1.8% |
| Ensemble (min N=3) | 100% | 100% | 100% | 0% | 2.6-3.3% | 1.6-1.8% |

**Takeaway:** Trivially solved by all SAC methods. PPO eureka fails even here. Eureka rewards consistently produce slightly higher jerk than vanilla (1.5-2× jerk violation rate) — early signal of aggressive optimization.

---

## Results: FetchPickAndPlace-v4 (Medium)

| Method | 100k | 250k | 500k | Hacking 100k | Hacking 250k | Hacking 500k |
|--------|------|------|------|-------------|-------------|-------------|
| Vanilla LLM | 2% | 10% | **46%** | 44% | 44% | **2%** |
| Eureka (PPO reward) | 2% | 12% | **6%** ⚠️ | 38% | 40% | **42%** 🎯 |
| Eureka (SAC+HER) | 3% | 17% | **49%** | 0% | 0% | 0% |
| Ensemble (min N=3) | 5% | 7% | **1%** ⚠️ | 0% | 0% | 0% |

**Safety metrics at 500k:**

| Method | Action Mag | Jerk | Action Viol | Jerk Viol | Drop Viol |
|--------|-----------|------|-------------|-----------|-----------|
| Vanilla LLM | 1.220 | 0.899 | 34.7% | 37.7% | 7.3% |
| Eureka (PPO reward) | 1.450 | 1.102 | 56.2% | 42.0% | 5.0% |
| Eureka (SAC+HER) | 1.033 | 0.706 | 17.7% | 30.6% | 4.0% |
| Ensemble (min N=3) | 0.987 | 0.018 | 5.2% | 0.0% | 0.0% |

**Key findings:**
1. **Eureka (PPO reward) = textbook Goodhart divergence.** Hacking increases with training (38→42%) and success *collapses* to 6%. The buggy HER wrapper made this worse but the pattern is real.
2. **Eureka (SAC+HER) = best method.** 49% success, 0% hacking, lower safety violations than vanilla.
3. **Vanilla LLM self-corrects.** Hacking drops from 44% to 2% by 500k — the agent finds real solutions eventually.
4. **Ensemble = too conservative.** 1% success. Zero violations because the robot barely moves. Min-aggregation killed learning.

---

## Results: FetchSlide-v4 (Hard)

| Method | 100k | 250k | 500k | Hacking 100k | Hacking 250k | Hacking 500k |
|--------|------|------|------|-------------|-------------|-------------|
| Vanilla LLM | 2% | 14% | 16% | 50% | 38% | 36% |
| Eureka (PPO reward) | 0% | 4% | **40%** | 46% | 48% | 2% |
| Eureka (SAC+HER) | 4% | 35% | **54%** | **46%** | **16%** | **0%** |
| Ensemble (min N=3) | 6% | 17% | **46%** | **44%** | **34%** | **7%** |

**Safety metrics at 500k:**

| Method | Action Mag | Jerk | Action Viol | Jerk Viol | Drop Viol |
|--------|-----------|------|-------------|-----------|-----------|
| Vanilla LLM | 1.192 | 0.333 | 17.2% | 8.8% | 41.9% |
| Eureka (PPO reward) | 1.264 | 0.460 | 24.5% | 14.5% | 42.6% |
| Eureka (SAC+HER) | 1.188 | 0.469 | 14.3% | 17.2% | 39.8% |
| Ensemble (min N=3) | 1.084 | 0.316 | 8.9% | 8.8% | 44.6% |

**Key findings:**
1. **Eureka (SAC+HER) = best performer** at 54% (+38% over vanilla). LLM reward shaping genuinely helps on hard tasks.
2. **Transient hacking arc (the money plot):** 46% → 16% → 0%. Agent exploits the proxy early, then finds real solutions.
3. **Safety during hacking:** Drop violations at 77% when hacking is 46% (100k), vs 40% when hacking resolves (500k).
4. **Vanilla LLM never recovers.** Hacking stays 36-50% — the vanilla reward is too weak to guide the agent.
5. **Eureka (PPO reward) eventually works** (40%) but took longer to overcome hacking.
6. **Ensemble works on Slide!** 46% success at 500k — unlike PickAndPlace where it killed learning. Hacking drops 44%→34%→7% (transient arc). Min-aggregation works when the task has clearer reward signal.

---

## Two Failure Modes of LLM Reward Overoptimization

### Mode A — Persistent Hacking (PickAndPlace, Eureka PPO reward)
- Hacking *increases* with training: 38% → 40% → 42%
- Success *decreases*: 12% → 6%
- Agent finds stable exploits and never escapes
- Safety degrades: action violations 33% → 56%

### Mode B — Transient Hacking (Slide, Eureka SAC+HER)
- Hacking *decreases* with training: 46% → 16% → 0%
- Success *increases*: 4% → 35% → 54%
- Agent exploits early but eventually finds genuine solutions
- **Problem:** Safety violations during the hacking phase are severe (77% drop violations)
- **Implication:** Even transient hacking is dangerous for real robots

---

## Mitigation 1: Ensemble (min, N=3) — Results

| Metric | Eureka single | Ensemble | Verdict |
|--------|--------------|----------|---------|
| Hacking (PickAndPlace) | 0% | 0% | ✅ Same |
| Success (PickAndPlace) | 49% | 1% | ❌ Killed learning |
| Safety (PickAndPlace) | Moderate | Perfect | ✅ But meaningless |
| Hacking (Reach) | 0% | 0% | ✅ Same |
| Success (Reach) | 100% | 100% | ✅ Same |
| Hacking (Slide) | 46%→0% | 44%→7% | ⚠️ Slower resolution |
| Success (Slide) | 54% | 46% | ⚠️ -8% but viable |
| Safety (Slide) | 14-17% viol | 8.9% action, 8.8% jerk | ✅ Better safety |

**Verdict:** Min-aggregation is task-dependent:
- **PickAndPlace:** Too conservative, kills learning (1% success)
- **Slide:** Works well! 46% success with better safety profile (8.9% vs 14.3% action violations)
- **Hypothesis:** Slide has a cleaner reward landscape — min-aggregation filters noise without destroying gradient. PickAndPlace requires more nuanced manipulation where conservative rewards starve exploration.

---

## Data Locations

```
generated_rewards/
├── *_vanilla.py                    Vanilla LLM rewards
├── eureka_ppo/                     Eureka + PPO iteration rewards
├── eureka_sac/                     Eureka + SAC+HER iteration rewards
└── ensemble/                       N=3 ensemble rewards per env

logs/sac_{100k,250k,500k}/          Old pipeline results (vanilla + eureka PPO reward)
scripts/results/eureka_sac/         New pipeline results (Eureka SAC+HER)
scripts/results/ensemble/           Ensemble results
results/hacking/                    Hacking rate analysis (old pipeline)
```

---

## Next Steps

| # | Task | Status | Deadline |
|---|------|--------|----------|
| 1 | Ensemble training — Slide | ✅ Done | May 28 |
| 2 | Try mean aggregation ensemble | TODO | May 30 |
| 3 | KL penalty mitigation (β sweep) | TODO | May 31 |
| 4 | Physics constraints mitigation | TODO | May 31 |
| 5 | Scaling plot (hacking + safety vs steps) | TODO | Jun 1 |
| 6 | Video recordings of hacking behavior | TODO | Jun 2 |
| 7 | **Poster** | TODO | **Jun 4** |
| 8 | **Final report** | TODO | **Jun 9** |
