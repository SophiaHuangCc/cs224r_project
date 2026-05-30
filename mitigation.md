# Mitigation Strategies for LLM Reward Hacking

> CS224R Final Project — Sunny Yuan & Sophia Huang
> Created: 2026-05-26

---

## Problem Statement

Our experiments show two failure modes of LLM-generated reward functions in robotic manipulation:

1. **Reward Hacking (Mode A)** — Agent exploits loopholes in the proxy reward to accumulate high reward without completing the task. (FetchPickAndPlace eureka: 8% success, 42% hacking rate at 500k steps)
2. **Unsafe Shortcut (Mode B)** — Agent finds aggressive strategies that succeed but violate safety constraints. (FetchSlide eureka: 48% success but 1.4-1.6× more safety violations)

Both stem from **overoptimization** of an imperfect proxy. The question: how do we maintain the sample-efficiency benefits of LLM reward shaping while preventing exploitation?

---

## Mitigation 1: Reward Ensemble (PRIMARY)

### Idea
Generate N independent LLM reward functions and aggregate them. A loophole exploitable in one reward is unlikely to exist across all N independently-generated functions.

### Mechanism
```
R_ensemble(s, a) = aggregate(r_1(s,a), r_2(s,a), ..., r_N(s,a))
```

Aggregation options:
- **Conservative (min):** `R = min(r_1, ..., r_N)` — most pessimistic, strongest anti-hacking
- **Mean:** `R = mean(r_1, ..., r_N)` — smooths out individual quirks
- **Trimmed mean:** Drop highest and lowest, average the rest

### Why It Addresses Hacking
- Hacking requires a systematic exploit in the reward landscape
- Independent reward functions create different landscapes with different exploits
- Taking the min/mean removes any single function's loopholes
- The agent can only get high aggregate reward by satisfying ALL reward functions → must solve the task genuinely

### Implementation
1. Call GPT-4o N times with varied prompts (different phrasings of the same task) or varied temperature (0.5, 0.7, 0.9, 1.1, 1.3)
2. Compile each into a callable `reward_fn_i`
3. Wrap in `EnsembleRewardWrapper` that computes all N rewards per step and returns the aggregate
4. Train SAC+HER as normal with the aggregated reward

### Ablation
- N = 1 (baseline, equivalent to vanilla LLM)
- N = 3
- N = 5

### Compute Cost
- Extra GPT-4o calls: N-1 per scenario (cheap, ~$0.01 each)
- Training time: Same as a single run (N reward forward passes per step are cheap numpy ops)
- Total: 2 envs × 1 training run each = 2 runs

### Expected Outcome
- Hacking rate ↓ (agent can't exploit a loophole that only exists in 1/N functions)
- Success rate ↑ or stable (genuine task completion is rewarded by all functions)
- Safety metrics may improve as side effect (can't use aggressive hacks)

### Literature
- Coste et al. ICLR 2024 — "Reward Model Ensembles Help Mitigate Overoptimization" (RLHF text domain)
- **Our contribution:** First application to LLM-generated reward functions in embodied/robotic domains

---

## Mitigation 2: KL Penalty Against Reference Policy

### Idea
Constrain the learned policy to stay close to a "reference" policy (trained with sparse reward), preventing it from drifting into weird exploit-seeking behavior.

### Mechanism
```
R_kl(s, a) = R_proxy(s, a) - β * log(π(a|s) / π_ref(a|s))
```

Where:
- `R_proxy` = LLM-generated reward
- `π_ref` = policy trained with sparse (ground-truth) reward
- `β` = KL penalty coefficient (hyperparameter)

### Why It Addresses Hacking
- Hacking requires the policy to deviate significantly from "normal" behavior
- The KL penalty makes it expensive to deviate from the reference
- Acts as a soft trust region: the agent can improve upon the reference but can't go to completely alien states/actions
- Directly analogous to PPO's KL penalty in RLHF (prevents reward model overoptimization in language models)

### Implementation
1. Train a reference SAC policy with sparse env reward (may already have this as baseline)
2. During training with LLM reward, at each step:
   - Forward pass through reference policy to get `π_ref(a|s)`
   - Forward pass through current policy to get `π(a|s)`
   - Compute KL term (for continuous actions: use Gaussian KL between the two policy distributions)
   - Modified reward: `r_modified = r_proxy - β * KL_step`
3. Tune β: too high → policy never improves beyond reference; too low → hacking returns

### Ablation
- β = 0 (no penalty, equivalent to pure Eureka)
- β = 0.01, 0.1, 1.0

### Compute Cost
- Pre-requisite: one sparse-reward training run per env (may already exist)
- Extra per step: one forward pass through frozen reference network (cheap)
- Total: 2 envs × 3 β values = 6 runs (or just pick one β)

### Expected Outcome
- Hacking rate ↓ (agent can't drift far from reference behavior)
- Success rate: bounded between reference and unconstrained proxy policy
- Trade-off: too high β = safe but slow; too low β = fast but hacks
- Sweet spot should give best of both worlds

### Literature
- Christiano et al. 2017 — Deep RL from Human Preferences
- Ouyang et al. 2022 — InstructGPT (KL penalty in RLHF)
- Gao et al. 2022 — Scaling Laws for Reward Model Overoptimization (shows KL budget controls Goodhart)
- **Our contribution:** Applying the RLHF KL-penalty framework to embodied LLM reward functions

---

## Mitigation 3: Reward Saturation (Bounded Proxy)

### Idea
Cap the maximum achievable proxy reward to remove the incentive for extreme exploitation. If reward is bounded, the gradient for "hack harder" vanishes once the cap is reached.

### Mechanism
```
R_sat(s, a) = tanh(R_proxy(s, a) / τ)
```
Or simpler:
```
R_sat(s, a) = clip(R_proxy(s, a), -C, +C)
```

Where τ (temperature) or C (cap) controls how aggressively we bound.

### Why It Addresses Hacking
- Hacking is profitable because the agent can get arbitrarily high proxy reward from an exploit
- With saturation, even the best exploit only gives reward ≤ C
- The marginal reward for "hacking harder" → 0 past the saturation point
- Makes the reward landscape flatter at extremes, reducing exploitation incentive
- Agent redirects optimization toward broader reward (which correlates with actual success) rather than maximizing one loophole

### Implementation
Literally one line in the reward wrapper:
```python
def compute_reward(self, achieved, desired, obs, action, info):
    raw = self.llm_reward_fn(achieved, desired, obs, action, info)
    return np.tanh(raw / self.tau)  # or np.clip(raw, -self.cap, self.cap)
```

### Ablation
- τ = ∞ or C = ∞ (no saturation, baseline)
- τ = 1.0, 5.0, 10.0 (or C = 1.0, 5.0, 10.0)
- Need to calibrate relative to the raw reward scale per scenario

### Compute Cost
- Zero extra compute (one numpy op per step)
- Total: 2 envs × 2-3 τ values = 4-6 runs

### Expected Outcome
- Hacking rate ↓ (exploit has diminishing returns)
- Success rate: may decrease slightly (reward signal is weaker overall)
- Risk: too aggressive saturation kills the learning signal entirely
- Best for: tasks where hacking involves extreme reward spikes (like FetchPickAndPlace eureka's lift bonus exploit)

### Literature
- Reward clipping is common in deep RL (DQN clipped rewards to [-1, 1])
- Conceptually related to reward normalization in PPO
- **Our contribution:** Framing reward saturation as an explicit anti-hacking tool for LLM-generated rewards, with controlled ablation

---

## Comparison

| Property | Ensemble | KL Penalty | Saturation |
|----------|----------|------------|------------|
| Addresses hacking? | ✅ Directly | ✅ Indirectly (constrains drift) | ✅ Partially (removes incentive) |
| Addresses unsafe behavior? | Maybe (side effect) | ✅ (stays near safe reference) | Weakly |
| Compute overhead | Negligible | One extra forward pass/step | Zero |
| Hyperparameters | N, aggregation method | β | τ or C |
| Prerequisites | None | Reference policy needed | Calibrate reward scale |
| Implementation complexity | Low | Medium | Very low |
| Risk of hurting learning | Low | High if β too large | High if τ too small |
| Novelty in embodied RL | High (first LLM ensemble for robots) | Medium (known technique, new context) | Low (known technique) |

---

## Recommended Plan

### If time allows only one:
→ **Ensemble (min, N=3)** on FetchPickAndPlace eureka. Compare hacking_rate before/after.

### If time allows two:
→ Ensemble + KL Penalty. Shows two complementary approaches (reward-side vs policy-side).

### Full comparison (ambitious but doable):
→ All three on FetchPickAndPlace (the clearest hacking case). Results table:

| Method | Success | Hacking Rate | Safety |
|--------|---------|--------------|--------|
| Vanilla LLM | 52% | 2% | moderate |
| Pure Eureka | 8% | 42% | bad |
| Eureka + Ensemble (N=3) | ? | ? | ? |
| Eureka + KL (β=0.1) | ? | ? | ? |
| Eureka + Saturation (τ=5) | ? | ? | ? |

---

## Mitigation 4: Physics-Informed Reward Verification (WAM-inspired)

### Inspiration
NVIDIA's World Action Models (WAMs) paradigm (arxiv 2605.12090, May 2026) and the Cosmos platform model the **joint distribution of future states AND actions**. Core insight: if you can predict what the world should look like after an action, you can **verify** whether a policy is genuinely solving the task vs gaming the reward.

### Idea
Train a lightweight forward dynamics model on FetchEnv transitions. Use it as a physics-plausibility verifier that penalizes trajectories the dynamics model finds surprising.

### Mechanism
```
f_dynamics: (s_t, a_t) → ŝ_{t+1}   (learned MLP forward model)
prediction_error = ||ŝ_{t+1} - s_{t+1}||_2

R_verified(s, a) = R_proxy(s, a) - λ * prediction_error
```

Or as a gate:
```
plausibility = exp(-prediction_error / σ)
R_verified(s, a) = R_proxy(s, a) * plausibility
```

### Why It Addresses Hacking
- Hacked behaviors exploit reward function loopholes that don't correspond to natural physics
- A learned dynamics model captures the **actual** state transitions of the environment
- If a policy achieves high proxy reward through physically implausible paths (e.g. jerky motions, teleporting objects, degenerate oscillations), the forward model's prediction error will be high
- High prediction error → penalty → hacking becomes unprofitable
- Genuine task completion follows natural dynamics → low prediction error → no penalty

### Implementation
1. Collect 50k-100k transitions from random/baseline policy in the Fetch env
2. Train a simple MLP: `(obs, action) → next_obs` (or just `(achieved_goal, action) → next_achieved_goal`)
3. During proxy-reward training, at each step:
   - Forward pass through dynamics model to get predicted next state
   - Compare against actual next state
   - Subtract scaled prediction error from proxy reward
4. The dynamics model is frozen during policy training (no adversarial games)

### Ablation
- λ = 0 (no verification, baseline)
- λ = 0.1, 1.0, 10.0
- Compare: MLP vs linear dynamics model (is nonlinearity needed?)

### Compute Cost
- Pre-train dynamics model: ~5 min (small MLP, 50k samples)
- Per-step overhead: one MLP forward pass (negligible)
- Total: 2 envs × 3 λ values = 6 runs

### Expected Outcome
- Hacking rate ↓ (physically implausible exploits get penalized)
- Success rate: should remain high (genuine solutions follow predictable dynamics)
- Side benefit: may also reduce safety violations (jerky/aggressive motions are less predictable)

### Literature
- Pathak et al. 2017 — Curiosity-Driven Exploration (prediction error as intrinsic reward, inverse application here)
- Hafner et al. 2020-2023 — Dreamer series (world models for model-based RL)
- NVIDIA Cosmos Reason (VLM with physics understanding for robot planning)
- WAMs survey (Wang et al. 2026) — unifying predictive state modeling with action generation
- **Our contribution:** Using a learned dynamics model as a physics-plausibility **penalty** to mitigate reward hacking (novel framing — existing work uses world models for planning/imagination, not as reward verifiers)

---

## Mitigation 5: Trajectory-Level KL with Physics Constraints (Cosmos Reason-inspired)

### Inspiration
Cosmos Reason combines prior knowledge, physics, and common sense to evaluate whether robot actions are sensible. We can approximate this: instead of just point-wise KL penalty (Mitigation 2), enforce **trajectory-level** consistency with learned physics.

### Idea
Train a small action-conditioned forward model and use state-consistency across a trajectory window as a reward multiplier. Hacked trajectories that achieve high reward but through inconsistent state sequences get suppressed.

### Mechanism
```
For trajectory window [t-k, ..., t]:
  consistency_score = mean(exp(-||f(s_i, a_i) - s_{i+1}||)) for i in window

R_consistent(s, a) = R_proxy(s, a) * consistency_score
```

### Why It Addresses Hacking
- Point-wise reward verification (Mitigation 4) can be fooled by individual timesteps
- Trajectory-level consistency catches **sequences** of implausible actions
- Reward hacking often manifests as temporally extended exploits (oscillation, repeated motions)
- A trajectory consistency gate makes it impossible to accumulate high reward through any sequence that violates physics

### Implementation
1. Same dynamics model as Mitigation 4
2. Maintain a rolling window of k=10 transitions
3. Compute consistency score over the window
4. Gate the proxy reward by consistency

### Compute Cost
- Same dynamics model (shared with Mitigation 4)
- k extra MLP forward passes per step (still negligible for k=10)

### Expected Outcome
- Strongest physics-based mitigation
- May be overkill for simple Fetch envs — best suited for longer-horizon tasks
- Nice theoretical narrative: "We bring world model verification to reward shaping"

### Literature
- Pan et al. 2022 — "The Effects of Reward Misspecification" (trajectory-level reward hacking)
- Cosmos Reason (NVIDIA, 2025) — physics + common sense for robot planning
- **Our contribution:** Trajectory-level physics consistency as an anti-hacking gate

---

## Mitigation 6: Physics-Constrained Reward Generation (LLM + Hard Constraints)

### Inspiration
Instead of verifying *after* the LLM generates rewards, constrain the LLM to generate rewards that are **inherently physics-aware**. Analogous to Constitutional AI but for physical laws.

### Idea
Add explicit physics constraints to the Eureka prompt, and post-process generated rewards to enforce physical plausibility bounds. The reward function itself cannot assign high reward to states that violate kinematics.

### Mechanism
```python
def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # LLM-generated task reward
    task_reward = ...  # distance-based shaping, etc.
    
    # Hard physics constraints (injected, not LLM-generated)
    velocity = np.linalg.norm(action[:3])  
    jerk = np.linalg.norm(action[:3] - prev_action[:3])
    
    physics_penalty = 0.0
    if velocity > MAX_SAFE_VELOCITY:
        physics_penalty -= (velocity - MAX_SAFE_VELOCITY) ** 2
    if jerk > MAX_SAFE_JERK:
        physics_penalty -= (jerk - MAX_SAFE_JERK) ** 2
    
    return task_reward + physics_penalty
```

### Why It Addresses Hacking
- Many reward hacks involve physically unreasonable behaviors (extreme actions, impossible object states)
- Hard-coding physics constraints into the reward makes these exploits impossible by construction
- The LLM handles task semantics; physics rules handle plausibility
- Separation of concerns: LLM ≠ physics engine

### Implementation
1. Identify key physics constraints for Fetch envs:
   - Max gripper velocity, max jerk
   - Object must remain on table (height > 0)
   - Object velocity/acceleration bounds
   - Gripper-object contact requirements for manipulation
2. Wrap LLM reward with hard constraint penalties
3. Constraints are NOT fed back to the LLM (to preserve the Eureka paper's methodology)
4. Compare: with/without physics constraints on same LLM reward

### Ablation
- No constraints (vanilla Eureka)
- Soft constraints (penalty terms)
- Hard constraints (reward = -∞ if violated, i.e. episode termination)

### Compute Cost
- Zero extra compute (few numpy ops per step)
- Implementation: ~30 lines of constraint code

### Expected Outcome
- Safety violations ↓ (by construction)
- Hacking rate ↓ (many hacks are physically unreasonable)
- Success rate: should be preserved or slightly lower (legitimate strategies sometimes push boundaries)
- Trade-off: too tight constraints → robot can't solve the task

### Literature
- Constrained MDPs (Altman 1999, Tessler et al. 2019)
- Safe RL with constraints (Dalal et al. 2018 — Safety Layer)
- **Our contribution:** Combining LLM reward generation (Eureka) with explicit physics constraint layers as a simple, zero-cost anti-hacking measure

---

## Updated Comparison

| Property | Ensemble | KL Penalty | Saturation | Physics Verifier | Traj Consistency | Physics Constraints |
|----------|----------|------------|------------|------------------|------------------|--------------------|
| Addresses hacking? | ✅ Directly | ✅ Indirectly | ✅ Partially | ✅ Directly | ✅ Strongly | ✅ By construction |
| Addresses unsafe behavior? | Maybe | ✅ | Weakly | ✅ | ✅ | ✅✅ (strongest) |
| Compute overhead | Negligible | One fwd pass/step | Zero | One MLP fwd/step | k MLP fwd/step | Zero |
| Hyperparameters | N, aggregation | β | τ or C | λ, model size | k, σ | Constraint thresholds |
| Prerequisites | None | Reference policy | Calibrate scale | Dynamics model | Dynamics model | Domain knowledge |
| Novelty | High | Medium | Low | **High** | High | Medium |
| WAM/Cosmos connection | — | — | — | ✅ (world model as verifier) | ✅ (trajectory physics) | ✅ (physics grounding) |

---

## Updated Recommended Plan

### For the poster (Jun 4) — pick 2-3 total:
1. **Ensemble (done ✅)** — results for Reach + PickAndPlace, Slide generating
2. **KL Penalty** — direct RLHF→robotics bridge (core project thesis)
3. **Physics Constraints** — zero-cost, easy to implement, strong safety story

### For the final report (Jun 9) — stretch:
4. **Physics Verifier (WAM-inspired)** — novel contribution, great narrative
5. Combine: Ensemble + Physics Constraints (belt + suspenders)

### Narrative for the paper:
> "We systematically study how techniques from LLM alignment (ensemble reward models, KL penalties) transfer to robotic reward overoptimization, and introduce a novel physics-informed reward verification approach inspired by World Action Models that catches reward hacking through dynamics prediction."

---

## Open Questions

1. For Ensemble: should we generate N rewards from the *same* iteration of Eureka, or run N independent Eureka loops? (Former is cheaper; latter gives more diversity) → **RESOLVED: using N independent Eureka loops (SAC+HER)**
2. For KL: do we have a good sparse-reward SAC baseline for PickAndPlace? (Need to check/train)
3. For Saturation: what's the right τ? Need to look at raw reward distributions from the 500k eval data.
4. Should we combine methods? (e.g., Ensemble + Saturation = belt and suspenders)
5. For Physics Verifier: how many transitions needed to train a good dynamics model for Fetch envs? (Hypothesis: 50k is enough given low-dim state space)
6. For Physics Constraints: what are the right constraint thresholds? (Check existing safety metrics code for values)
