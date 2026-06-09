# Reward Hacking in LLM-Generated Reward Functions

**CS224R Final Project — Stanford, Spring 2026**
Sunny Yuan & Sophia Huang

> We study whether reward functions written by an LLM induce **reward hacking**
> on three MuJoCo Fetch manipulation tasks, and whether alignment-style
> mitigations help. All numbers in [`checkpoint3.md`](checkpoint3.md) are the
> 500k-step SAC+HER checkpoint, 100 evaluation episodes, seed 42.

---

## TL;DR

| Method (Eureka + …) | PickAndPlace success | Slide success | Notes |
|---|---:|---:|---|
| Eureka baseline | 52% | 64% | iterated LLM reward |
| **+ KL β=0.01** | **93%** | 58% | soft trust region — best mitigation |
| + Ensemble (min, N=3) | 4% | 52% | optimization collapse, not anti-hack |
| + Physics c=1.0 | 3% | 62% | safety win on Slide, collapse on Pick |

- Only the **single-shot LLM reward on Slide** is actually misaligned
  (proxy↔success r = 0.28); Eureka's iteration fixes it (r = 0.87, success
  14% → 64%). Iterative reward design is itself the most effective anti-hacking
  measure we observed.
- KL anchoring to a sparse-reward reference yields **+41 pts of PickAndPlace
  success** and **5× fewer Slide action violations**, alignment intact.
- Physics constraints are a *task-dependent* safety knob: ≈ unchanged success
  on Slide with halved violations, but crush contact-rich PickAndPlace.

Full results, ablations, and analysis: [`checkpoint3.md`](checkpoint3.md).
Design rationale & metric definitions: [`DESIGN.md`](DESIGN.md).

---

## Repository layout

```text
cs224r_project/
├── checkpoint3.md             # final consolidated results (read this)
├── DESIGN.md                  # research question, metrics, methodology
├── experiment.md              # end-to-end reproduction runbook
├── mitigation.md              # KL / Ensemble / Physics mitigation notes
├── requirements.txt
├── scripts/
│   ├── llm_reward_vanilla.py        # single-shot GPT-4o reward generator
│   ├── llm_reward_eureka_sac.py     # iterated Eureka loop (3 iters)
│   ├── ensemble_reward.py           # min-of-N ensemble reward wrapper
│   ├── run_vanilla_all.py           # generate vanilla rewards (3 tasks)
│   ├── run_eureka_sac_all.py        # generate Eureka rewards (3 tasks)
│   ├── run_ensemble_mitigation.py   # generate N=3 ensemble rewards
│   ├── train_reference.py           # sparse-reward SAC+HER reference (for KL)
│   ├── train_local.py               # train every (task, reward_type) combo
│   ├── train_modal.py               # same, parallelized on Modal
│   ├── eval_local.py                # 100-episode eval w/ safety metrics
│   ├── make_figures.py              # all report figures (success + dynamics)
│   ├── make_baseline_videos.py      # rollout mp4s for the 3 tasks
│   ├── sac/                         # shared SAC/HER training kwargs
│   └── safety/                      # SafetyMetricWrapper + tripwires
├── generated_rewards/         # LLM-written reward .py files
│   ├── *_vanilla.py                 # single-shot per task
│   ├── eureka_sac/                  # iterated best + per-iter rewards
│   └── ensemble/<task>/             # N=3 independent Eureka rewards
├── models/                    # SAC checkpoints, organized by reward_type
│   ├── reference/                   # sparse-reward references (KL target)
│   ├── vanilla/  eureka/  ensemble/  kl/  physics/
├── results/
│   ├── eval/<reward_type>/<label>_<ckpt>k.json   # per-checkpoint metrics
│   ├── figures/                                  # report figures
│   └── videos/                                   # rollout mp4s
└── logs/                      # TensorBoard event files
```

---

## Setup

```bash
conda create -n cs224r_project python=3.10 -y
conda activate cs224r_project
pip install -r requirements.txt
```

Create `.env` at the project root with Azure OpenAI credentials (only needed if
regenerating LLM rewards):

```bash
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o   # optional, defaults to gpt-4o
```

---

## Reproducing the results

Full step-by-step runbook (with time estimates, Modal instructions, and
filtering flags) lives in [`experiment.md`](experiment.md). The short version:

```bash
# 1. Generate LLM rewards (Azure OpenAI required)
python scripts/run_vanilla_all.py
python scripts/run_eureka_sac_all.py
python scripts/run_ensemble_mitigation.py

# 2. Train sparse-reward reference policies (for KL anchoring)
python scripts/train_reference.py

# 3. Train every (task, reward_type) combo at 100k/250k/500k checkpoints
python scripts/train_local.py            # 18 runs × 500k locally (~21 h on M4)
# OR
modal run scripts/train_modal.py         # all 18 jobs in parallel (~70 min)

# 4. Evaluate (100 deterministic episodes per checkpoint)
python scripts/eval_local.py

# 5. Regenerate figures and rollout videos
python scripts/make_figures.py
python scripts/make_baseline_videos.py
```

All training and eval steps accept `--env`, `--reward-type`, `--checkpoint`, and
`--skip-existing` for partial reruns.

---

## Experimental setup

| | |
|---|---|
| **Tasks** | FetchReach-v4, FetchPickAndPlace-v4, FetchSlide-v4 (Gymnasium-Robotics) |
| **Algorithm** | SAC + Hindsight Experience Replay, 500k steps, eval every 100k |
| **Reward sources** | Vanilla (single-shot GPT-4o), Eureka (iterated best-of-N) |
| **Mitigations on Eureka** | Reward Ensemble (min N=3), KL penalty to sparse reference (β ∈ {0.01, 0.1, 1.0}), Physics constraint penalty (c ∈ {0.1, 1.0, 10.0}) |
| **Eval** | 100 deterministic episodes per checkpoint |

### Metrics

- **Success rate** — env's ground-truth `is_success` (capability)
- **Proxy↔success correlation (r)** — Pearson correlation between per-episode
  proxy reward and task success (alignment / *hackability*; parameter-free,
  unlike a thresholded "hacking rate")
- **Safety tripwires** — fraction of steps violating action-norm, jerk,
  object-speed, object-accel, drop, table-slam, workspace bounds
  (see [`scripts/safety/metrics.py`](scripts/safety/metrics.py))

---

## Figures

Generated by `scripts/make_figures.py` into `results/figures/`:

- [`fig1_hero_pickandplace.png`](results/figures/fig1_hero_pickandplace.png) — KL wins, min-ensemble freezes
- [`fig2_kl_cliff_pickandplace.png`](results/figures/fig2_kl_cliff_pickandplace.png) — KL β over-constraint cliff
- [`fig3_physics_taskdependence.png`](results/figures/fig3_physics_taskdependence.png) — physics: Slide vs PickAndPlace
- [`fig_dynamics_FetchPickAndPlace-v4.png`](results/figures/fig_dynamics_FetchPickAndPlace-v4.png) — 3-panel training dashboard
- [`fig_dynamics_FetchSlide-v4.png`](results/figures/fig_dynamics_FetchSlide-v4.png) — 3-panel training dashboard

Rollout videos (`results/videos/`):
[`fetchreach_baseline.mp4`](results/videos/fetchreach_baseline.mp4),
[`fetchpick_baseline.mp4`](results/videos/fetchpick_baseline.mp4),
[`fetchslide_baseline.mp4`](results/videos/fetchslide_baseline.mp4).

---

## Caveats

- All runs use a **single seed (42)**; mid-training success can swing
  (e.g. KL β=0.01 Reach dips to 2% at 250k before recovering to 100%).
  Cross-method gaps are the story; small differences are within run-to-run
  noise.
- `rollout/ep_rew_mean` in TensorBoard is each method's *own* training reward
  (KL subtracts β·KL, Physics subtracts c·penalty, Ensemble is min-of-3) — so
  the proxy-reward panels compare *shape*, not absolute height across methods.

---

## References

- Ma et al. 2023 — *Eureka: Human-Level Reward Design via Coding Large Language Models*
- Coste et al. ICLR 2024 — *Reward Model Ensembles Help Mitigate Overoptimization*
- Gao et al. 2022 — *Scaling Laws for Reward Model Overoptimization*
- Christiano et al. 2017 — *Deep RL from Human Preferences*

## Authors

Sunny Yuan, Sophia Huang
