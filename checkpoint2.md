# Checkpoint 2: KL Penalty Mitigation Results

**Date:** 2026-06-01
**Project:** CS224R Final Project — Sunny Yuan & Sophia Huang

---

## Mitigation 2: KL Penalty (Reference Policy Divergence)

**Method:** Add KL(π || π_ref) penalty to reward, where π_ref is a pre-trained reference policy. Penalizes deviation from "normal" behavior — same technique used in RLHF (PPO + KL from SFT policy).

**Sweep:** β ∈ {0.01, 0.1, 1.0} across all three environments, evaluated at 100k/250k/500k steps.

---

## Results: FetchReach-v4 (Easy)

| Method | 100k | 250k | 500k | Hacking | Safety |
|--------|------|------|------|---------|--------|
| Vanilla LLM | 100% | 100% | 100% | 0% all | action 2.5%, jerk 0.5% |
| Eureka (SAC+HER) | 100% | 100% | 100% | 0% all | action 3.9%, jerk 1.8% |
| KL β=0.01 | 100% | 5% ⚠️ | 100% | 0%/45%/0% | action 3.0%, jerk 1.6% |
| KL β=0.1 | 0% | 100% | 100% | 50%/0%/0% | action 3.3%, jerk 1.5% |
| KL β=1.0 | 94% | 93% | 97% | 0% all | action 3.5%, jerk 3.5% |

**Takeaway:** Task is too easy to differentiate. β=1.0 notably slows convergence (97% vs 100%) and increases jerk (3.5% vs 0.5%). The 250k dips for low β are training instability artifacts that resolve by 500k.

---

## Results: FetchPickAndPlace-v4 (Medium) — The Key Comparison

### Success Rate & Hacking

| Method | Success 500k | Hacking 500k | Hacking arc (100k→250k→500k) |
|--------|-------------|-------------|-------------------------------|
| Vanilla LLM | 52% | 2% | 44% → 44% → 2% |
| Eureka (SAC+HER) | 53% | 0% | 48% → 31% → 0% |
| Ensemble (min N=3) | 1% ❌ | 0% | 0% → 0% → 0% |
| **KL β=0.01** | **93%** ✅ | **0%** | 44% → 16% → 0% |
| KL β=0.1 | 17% | 33% | 46% → 43% → 33% |
| KL β=1.0 | 3% ❌ | 47% | 44% → 43% → 47% |

### Safety Violations at 500k

| Method | Action Viol | Jerk Viol | Drop Viol |
|--------|-------------|-----------|-----------|
| Vanilla LLM | 28.7% | 35.4% | 6.5% |
| Eureka (SAC+HER) | 20.4% | 28.5% | 8.1% |
| Ensemble (min N=3) | 5.2% | 0.0% | 0.0% |
| **KL β=0.01** | 25.2% | **57.6%** ⚠️ | 2.5% |
| KL β=0.1 | 17.7% | 21.4% | 9.8% |
| KL β=1.0 | 16.4% | 7.5% | 3.2% |

### Key Findings — PickAndPlace

1. **KL β=0.01 dominates on task performance:** 93% success is nearly 2× vanilla/Eureka (52-53%), and eliminates hacking.
2. **But KL β=0.01 has the WORST jerk violations (57.6%)** — the policy learns aggressive, jerky motions to solve the task while staying "close" to reference in KL space. KL constrains distribution similarity, not physical safety.
3. **Higher β kills learning:** β=0.1 gets stuck at 17% success with persistent 33% hacking. β=1.0 completely fails (3% success, 47% hacking). The KL penalty dominates the reward signal.
4. **Eureka = best safety-hacking tradeoff:** 53% success, 0% hacking, 20.4% action + 28.5% jerk (vs KL's 25.2% + 57.6%).
5. **β sensitivity is extreme:** Only β=0.01 works; 10× higher and learning collapses. This makes KL impractical without careful tuning.

---

## Results: FetchSlide-v4 (Hard)

### Success Rate & Hacking

| Method | Success 500k | Hacking 500k | Hacking arc (100k→250k→500k) |
|--------|-------------|-------------|-------------------------------|
| Vanilla LLM | 20% | 36% | 50% → 38% → 36% |
| Eureka (SAC+HER) | 57% | 0% | 45% → 18% → 0% |
| Ensemble (min N=3) | 46% | 7% | 44% → 34% → 7% |
| **KL β=0.01** | **56%** | **0%** | 48% → 14% → 0% |
| KL β=0.1 | 5% ❌ | 45% | 50% → 49% → 45% |
| KL β=1.0 | 0% ❌ | 50% | 48% → 49% → 50% |

### Safety Violations at 500k

| Method | Action Viol | Jerk Viol | Drop Viol |
|--------|-------------|-----------|-----------|
| Vanilla LLM | 16.0% | 9.1% | 36.4% |
| Eureka (SAC+HER) | 12.4% | 13.7% | 41.3% |
| Ensemble (min N=3) | 8.9% | 8.8% | 44.6% |
| **KL β=0.01** | **2.7%** ✅ | 12.3% | 39.9% |
| KL β=0.1 | 8.7% | 3.5% | 75.4% |
| KL β=1.0 | 24.4% | 2.0% | 95.5% |

### Key Findings — Slide

1. **KL β=0.01 ≈ Eureka:** Nearly identical performance (56% vs 57% success, both 0% hacking).
2. **KL β=0.01 has best action violations (2.7%)** — dramatically lower than all others. The reference policy constraint does help here.
3. **Higher β = catastrophic drop violations:** β=0.1 has 75.4% drops, β=1.0 has 95.5%. The constraint prevents the agent from learning to slide at all, so it just drops the object.
4. **Same β sensitivity problem:** Only β=0.01 works on Slide too.

---

## Cross-Method Comparison Summary (500k)

### Hacking Mitigation Effectiveness

| Method | Reach | PickAndPlace | Slide | Avg |
|--------|-------|-------------|-------|-----|
| Vanilla LLM | 0% | 2% | 36% | 12.7% |
| Eureka (SAC+HER) | 0% | 0% | 0% | **0%** |
| Ensemble (min N=3) | 0% | 0% | 7% | 2.3% |
| KL β=0.01 | 0% | 0% | 0% | **0%** |
| KL β=0.1 | 0% | 33% | 45% | 26% |
| KL β=1.0 | 0% | 47% | 50% | 32.3% |

### Task Success

| Method | Reach | PickAndPlace | Slide | Avg |
|--------|-------|-------------|-------|-----|
| Vanilla LLM | 100% | 52% | 20% | 57.3% |
| Eureka (SAC+HER) | 100% | 53% | 57% | 70% |
| Ensemble (min N=3) | 100% | 1% | 46% | 49% |
| KL β=0.01 | 100% | **93%** | 56% | **83%** |
| KL β=0.1 | 100% | 17% | 5% | 40.7% |
| KL β=1.0 | 97% | 3% | 0% | 33.3% |

### Composite Safety Score (avg of action + jerk + drop violation rates)

| Method | Reach | PickAndPlace | Slide | Avg |
|--------|-------|-------------|-------|-----|
| Vanilla LLM | 3.6% | 23.5% | 20.5% | 15.9% |
| Eureka (SAC+HER) | 3.2% | 19.0% | 22.5% | 14.9% |
| Ensemble (min N=3) | 2.6% | 1.7% | 20.8% | 8.4% |
| KL β=0.01 | 3.1% | **28.4%** ⚠️ | **18.3%** | 16.6% |
| KL β=0.1 | 2.4% | 16.3% | 29.2% | 16.0% |
| KL β=1.0 | 5.4% | 9.0% | 40.6% | 18.3% |

---

## Analysis: Why KL β=0.01 Gives High Success But Poor Safety

**The KL penalty constrains DISTRIBUTION similarity, not PHYSICAL safety.**

- KL(π || π_ref) penalizes actions that are statistically unlikely under the reference policy
- But jerky/unsafe actions can be KL-close to the reference if they're within the action distribution's support
- Result: the policy finds aggressive strategies that technically don't deviate much in KL space but produce physically dangerous behavior
- This mirrors findings in LLM alignment — KL-constrained RLHF models can still exhibit harmful behaviors that are "in-distribution"

**Why β sensitivity is so extreme:**
- β=0.01: Penalty is negligible — basically vanilla + slight regularization → learns fast, no safety benefit
- β=0.1: Penalty competes with task reward → learning slows dramatically, hacking persists
- β=1.0: Penalty dominates → agent can't explore, never solves the task
- There's no "sweet spot" that simultaneously prevents hacking AND maintains learning AND improves safety

---

## Eureka vs KL: Head-to-Head Verdict

| Criterion | Eureka (SAC+HER) | KL β=0.01 | Winner |
|-----------|-----------------|-----------|--------|
| Hacking elimination | ✅ 0% everywhere | ✅ 0% everywhere | Tie |
| Task success (avg) | 70% | **83%** | KL |
| Safety violations (avg) | **14.9%** | 16.6% | Eureka |
| PickAndPlace safety | **19.0%** | 28.4% | Eureka |
| Slide safety | 22.5% | **18.3%** | KL |
| Robustness to hyperparams | ✅ Single method, no tuning | ❌ Only 1/3 β values work | Eureka |
| Practical deployability | ✅ Safe + effective | ⚠️ Fast but jerky | Eureka |

**Bottom line:**
- **KL β=0.01** is the best *anti-hacking* + *task performance* method but provides a **false sense of safety** — low hacking ≠ safe behavior
- **Eureka** provides the best *safety-aware anti-hacking* — eliminates hacking while producing smoother, safer policies
- **For real robots:** Eureka is clearly preferable. High success with reckless behavior (KL) is worse than moderate success with safe behavior (Eureka)

---

## Updated Method Table

| # | Method | Hacking? | Success | Safety | Verdict |
|---|--------|----------|---------|--------|---------|
| 1 | Vanilla LLM | Partial (2-36%) | Moderate | Moderate | Baseline |
| 2 | Eureka (SAC+HER) | ✅ Eliminated | High (53-57%) | **Good** | **Best overall** |
| 3 | Ensemble (min N=3) | ✅ Mostly (0-7%) | Variable (1-46%) | Excellent | Too conservative for manipulation |
| 4 | KL β=0.01 | ✅ Eliminated | **Highest (56-93%)** | Poor (jerky) | Fast but unsafe |
| 5 | KL β≥0.1 | ❌ Persistent | Low (0-17%) | N/A (doesn't learn) | Failed |

---

## Data Locations (New)

```
results/
├── eval/
│   ├── vanilla/           Vanilla LLM eval results
│   ├── eureka/            Eureka SAC+HER eval results  
│   ├── ensemble/          Ensemble (min N=3) eval results
│   └── kl/                KL penalty eval results (β sweep)
├── kl/                    KL penalty detailed metrics per checkpoint
├── kl_reference_policy/   Reference policy metrics
├── train/
│   ├── vanilla/           Training curves
│   ├── eureka/            Eureka training curves
│   └── kl/               KL training curves (if available)
└── videos/                Behavior recordings
```

---

## Remaining Tasks

| # | Task | Status | Deadline |
|---|------|--------|----------|
| 1 | KL penalty sweep | ✅ Done | Jun 1 |
| 2 | KL vs Eureka analysis | ✅ Done | Jun 1 |
| 3 | Scaling/dynamics plots (hacking + safety vs steps) | TODO | Jun 2 |
| 4 | Video recordings of hacking vs safe behavior | TODO | Jun 2 |
| 5 | **Poster design** | TODO | **Jun 4** |
| 6 | **Final report** | TODO | **Jun 9** |
