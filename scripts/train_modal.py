"""
Run all SAC+HER training scenarios in parallel on Modal.
Mirrors train_local.py — one Modal job per (task, reward_type[, kl_beta]) combination.

For reward type "kl", spawns one job per β value in KL_BETAS (so 3 jobs/task).
Requires reference models in models/reference/ (run train_reference.py locally first).

Usage:
    # Run all scenarios
    modal run scripts/train_modal.py

    # Filter by task or reward type
    modal run scripts/train_modal.py --env FetchReach-v4
    modal run scripts/train_modal.py --reward-type kl

Results are stored in the Modal volume and can be downloaded with:
    modal volume get cs224r-sac-results /results/results ./results/
    modal volume get cs224r-sac-results /results/models  ./models/
    modal volume get cs224r-sac-results /results/logs    ./logs/
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

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
    .add_local_dir(str(project_root / "models" / "reference"), remote_path="/root/project/models/reference")
)

vol = modal.Volume.from_name("cs224r-sac-results", create_if_missing=True)

# ── Experiment config (mirrors train_local.py) ───────────────────────────────

TASKS = ["FetchReach-v4", "FetchPickAndPlace-v4", "FetchSlide-v4"]
CHECKPOINTS = [100_000, 250_000, 500_000]
TIMESTEPS = 500_000
SEED = 42
EVAL_EPISODES = 100
REWARD_TYPES = ["vanilla", "eureka", "ensemble", "kl"]
KL_BETAS = [0.01, 0.1, 1.0]


def reward_config(task: str, reward_type: str) -> dict:
    if reward_type == "vanilla":
        return {"paths": [f"generated_rewards/{task}_vanilla.py"]}
    if reward_type in ("eureka", "kl"):
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
def train_scenario(task: str, reward_type: str, kl_beta: Optional[float] = None) -> dict:
    import sys
    sys.path.insert(0, "/root/project/scripts")
    os.chdir("/root/project")

    from stable_baselines3 import SAC

    from sac import (
        load_reward_fn_from_file,
        train_with_checkpoints,
        train_with_kl_checkpoints,
    )
    from ensemble_reward import make_ensemble_reward_fn

    cfg = reward_config(task, reward_type)
    paths = cfg["paths"]
    if len(paths) == 1:
        reward_fn = load_reward_fn_from_file(paths[0])
    else:
        fns = [load_reward_fn_from_file(p) for p in paths]
        reward_fn = make_ensemble_reward_fn(fns, aggregation=cfg.get("aggregation", "min"))

    if reward_type == "kl":
        assert kl_beta is not None, "kl reward_type requires kl_beta"
        label = f"{task}_kl_b{kl_beta}"
        ref_path = f"models/reference/{task}.zip"
        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"Reference policy not found: {ref_path}. "
                f"Train it locally first with `python scripts/train_reference.py --env {task}` "
                f"and rerun this Modal job — train_modal.py uploads models/reference/."
            )
        reference_model = SAC.load(ref_path)

        print(f"Starting {label}...")
        all_metrics = train_with_kl_checkpoints(
            env_id=task,
            reward_fn=reward_fn,
            reference_model=reference_model,
            kl_beta=kl_beta,
            timesteps=TIMESTEPS,
            checkpoints=CHECKPOINTS,
            seed=SEED,
            eval_episodes=EVAL_EPISODES,
            save_dir="/results/models/kl",
            results_dir="/results/results/kl",
            label=label,
            tensorboard_log=f"/results/logs/kl/{label}",
        )
    else:
        label = f"{task}_{reward_type}"
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

def _expand_scenarios(tasks: list[str], reward_types: list[str]) -> list[tuple]:
    """Expand to one (task, reward_type, kl_beta_or_None) per Modal job."""
    out = []
    for t in tasks:
        for rt in reward_types:
            if rt == "kl":
                for beta in KL_BETAS:
                    out.append((t, rt, beta))
            else:
                out.append((t, rt, None))
    return out


@app.local_entrypoint()
def main(env: str = "all", reward_type: str = "all"):
    import time

    tasks = TASKS if env == "all" else [env]
    reward_types = REWARD_TYPES if reward_type == "all" else [reward_type]
    scenarios = _expand_scenarios(tasks, reward_types)

    print(f"Launching {len(scenarios)} job(s) on Modal...")
    print(f"  Tasks:        {tasks}")
    print(f"  Reward types: {reward_types}")
    print(f"  Checkpoints:  {[f'{c//1000}k' for c in CHECKPOINTS]}")
    if "kl" in reward_types:
        print(f"  KL β sweep:   {KL_BETAS}")
    print()

    start = time.time()
    futures = [train_scenario.spawn(t, rt, beta) for t, rt, beta in scenarios]

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
