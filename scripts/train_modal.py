"""
Run all SAC+HER training scenarios in parallel on Modal.
Mirrors train_local.py — one Modal job per (task, reward_type) combination.

Usage:
    # Run all 9 scenarios
    modal run scripts/train_modal.py

    # Filter by task or reward type
    modal run scripts/train_modal.py --env FetchReach-v4
    modal run scripts/train_modal.py --reward-type eureka

Results are stored in the Modal volume and can be downloaded with:
    modal volume get cs224r-sac-results /results/results ./results/
    modal volume get cs224r-sac-results /results/models  ./models/
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

# ── Modal setup ──────────────────────────────────────────────────────────────

app = modal.App("cs224r-sac-training")
project_root = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "numpy", "gymnasium", "gymnasium-robotics",
        "stable-baselines3", "mujoco", "torch", "tensorboard",
    )
    .add_local_dir(str(project_root / "scripts"), remote_path="/root/project/scripts")
    .add_local_dir(str(project_root / "generated_rewards"), remote_path="/root/project/generated_rewards")
)

vol = modal.Volume.from_name("cs224r-sac-results", create_if_missing=True)

# ── Experiment config (mirrors train_local.py) ───────────────────────────────

TASKS = ["FetchReach-v4", "FetchPickAndPlace-v4", "FetchSlide-v4"]
CHECKPOINTS = [100_000, 250_000, 500_000]
TIMESTEPS = 500_000
SEED = 42
EVAL_EPISODES = 100
REWARD_TYPES = ["vanilla", "eureka", "ensemble"]


def reward_config(task: str, reward_type: str) -> dict:
    if reward_type == "vanilla":
        return {"paths": [f"generated_rewards/{task}_vanilla.py"]}
    if reward_type == "eureka":
        return {"paths": [f"generated_rewards/eureka_sac/{task}_best.py"]}
    if reward_type == "ensemble":
        return {
            "paths": [f"generated_rewards/ensemble/{task}/reward_{i}.py" for i in range(1, 4)],
            "aggregation": "min",
        }
    raise ValueError(f"Unknown reward type: {reward_type!r}")


# ── Modal training function ──────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={"/results": vol},
    timeout=18_000,  # 5 hours — enough for 500k steps
    cpu=4,
    memory=8192,
)
def train_scenario(task: str, reward_type: str) -> dict:
    import sys
    sys.path.insert(0, "/root/project/scripts")
    os.chdir("/root/project")

    from sac import load_reward_fn_from_file, train_with_checkpoints
    from ensemble_reward import make_ensemble_reward_fn

    label = f"{task}_{reward_type}"
    cfg = reward_config(task, reward_type)

    if len(cfg["paths"]) == 1:
        reward_fn = load_reward_fn_from_file(cfg["paths"][0])
    else:
        fns = [load_reward_fn_from_file(p) for p in cfg["paths"]]
        reward_fn = make_ensemble_reward_fn(fns, aggregation=cfg.get("aggregation", "min"))

    print(f"Starting {label}...")
    all_metrics = train_with_checkpoints(
        env_id=task,
        reward_fn=reward_fn,
        timesteps=TIMESTEPS,
        checkpoints=CHECKPOINTS,
        seed=SEED,
        eval_episodes=EVAL_EPISODES,
        save_dir=f"/results/models/{reward_type}",
        results_dir=f"/results/results/{reward_type}",
        label=label,
        tensorboard_log=f"/results/logs/{reward_type}/{label}",
    )
    vol.commit()

    best_ckpt = max(all_metrics, key=lambda c: all_metrics[c]["success_rate"])
    best_sr = all_metrics[best_ckpt]["success_rate"]
    print(f"Done: {label} — best success={best_sr:.1%} @ {best_ckpt//1000}k")
    return {"label": label, "best_checkpoint": best_ckpt, "metrics": all_metrics}


# ── Entry point ───────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(env: str = "all", reward_type: str = "all"):
    import time

    tasks = TASKS if env == "all" else [env]
    reward_types = REWARD_TYPES if reward_type == "all" else [reward_type]
    scenarios = [(t, r) for t in tasks for r in reward_types]

    print(f"Launching {len(scenarios)} job(s) on Modal...")
    print(f"  Tasks:        {tasks}")
    print(f"  Reward types: {reward_types}")
    print(f"  Checkpoints:  {[f'{c//1000}k' for c in CHECKPOINTS]}\n")

    start = time.time()
    futures = [train_scenario.spawn(t, r) for t, r in scenarios]

    for future in futures:
        result = future.get()
        best_m = result["metrics"][result["best_checkpoint"]]
        print(f"  Done: {result['label']} — success={best_m['success_rate']:.1%} "
              f"@ {result['best_checkpoint']//1000}k")

    elapsed = time.time() - start
    print(f"\nAll {len(scenarios)} job(s) finished in {elapsed/60:.0f}min")
    print("\nTo download results:")
    print("  modal volume get cs224r-sac-results /results/results ./results/")
    print("  modal volume get cs224r-sac-results /results/models  ./models/")
    print("  modal volume get cs224r-sac-results /results/logs    ./logs/")
    print("Then: tensorboard --logdir logs/")
