"""
Modal-based parallel SAC+HER training for all scenarios and step counts.
Runs 6 scenarios × 3 step counts = 18 jobs in parallel on Modal.

Usage:
    cd cs224r_project
    conda run -n cs224r_project modal run scripts/sac/train_modal.py

Results are downloaded to logs/sac_{steps}/ locally.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("cs224r-sac-training")

project_root = Path(__file__).resolve().parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "numpy",
        "gymnasium",
        "gymnasium-robotics",
        "stable-baselines3",
        "mujoco",
        "torch",
        "tensorboard",
        "openai",
    )
    .add_local_dir(
        str(project_root / "scripts"),
        remote_path="/root/project/scripts",
    )
    .add_local_dir(
        str(project_root / "generated_rewards"),
        remote_path="/root/project/generated_rewards",
    )
)

# Volume for persisting results
vol = modal.Volume.from_name("cs224r-sac-results", create_if_missing=True)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
SCENARIOS = [
    ("FetchReach-v4", "generated_rewards/FetchReach-v4_vanilla.py", "FetchReach_vanilla"),
    ("FetchReach-v4", "generated_rewards/eureka_ppo/FetchReach-v4_iter1.py", "FetchReach_eureka"),
    ("FetchPickAndPlace-v4", "generated_rewards/FetchPickAndPlace-v4_vanilla.py", "FetchPickAndPlace_vanilla"),
    ("FetchPickAndPlace-v4", "generated_rewards/eureka_ppo/FetchPickAndPlace-v4_best.py", "FetchPickAndPlace_eureka"),
    ("FetchSlide-v4", "generated_rewards/FetchSlide-v4_vanilla.py", "FetchSlide_vanilla"),
    ("FetchSlide-v4", "generated_rewards/eureka_ppo/FetchSlide-v4_best.py", "FetchSlide_eureka"),
]

STEP_COUNTS = [250_000, 500_000]


def steps_label(steps: int) -> str:
    if steps >= 1_000_000:
        return f"{steps // 1_000_000}M"
    return f"{steps // 1000}k"


# ---------------------------------------------------------------------------
# Training function (runs on Modal)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/results": vol},
    timeout=14400,  # 4 hours max per job
    cpu=4,
    memory=8192,
)
def train_scenario(env_id: str, reward_path: str, label: str, steps: int, seed: int = 42):
    """Train a single SAC+HER scenario on Modal."""
    import sys
    sys.path.insert(0, "/root/project/scripts")
    os.chdir("/root/project")

    from sac import load_reward_fn_from_file, train_sac_her_with_reward

    slabel = steps_label(steps)
    model_dir = f"/results/models/sac_{slabel}"
    log_dir = f"/results/logs/sac_{slabel}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    save_path = f"{model_dir}/{label}"
    metrics_path = f"{log_dir}/{label}_metrics.json"

    print(f"🚀 Training {label} @ {slabel} ({steps} steps)")
    print(f"   env={env_id}  reward={reward_path}")

    reward_fn = load_reward_fn_from_file(reward_path)

    _, metrics = train_sac_her_with_reward(
        env_id=env_id,
        reward_fn=reward_fn,
        timesteps=steps,
        seed=seed,
        eval_episodes=50,
        save_path=save_path,
        verbose=1,
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ {label} @ {slabel} done — success_rate={metrics.get('success_rate', 'N/A')}")
    vol.commit()

    return {"label": label, "steps": steps, "metrics": metrics}


# ---------------------------------------------------------------------------
# Retrieve metrics from volume
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={"/results": vol}, timeout=300)
def get_metrics(step_label: str):
    """Read all metrics for a given step count."""
    log_dir = f"/results/logs/sac_{step_label}"
    results = {}
    if os.path.exists(log_dir):
        for f in sorted(os.listdir(log_dir)):
            if f.endswith("_metrics.json"):
                with open(os.path.join(log_dir, f)) as fh:
                    label = f.replace("_metrics.json", "")
                    results[label] = json.load(fh)
    return results


# ---------------------------------------------------------------------------
# Entrypoint: launch all 18 jobs in parallel
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    import time

    print(f"Launching {len(SCENARIOS) * len(STEP_COUNTS)} training jobs on Modal...")
    print(f"Scenarios: {[s[2] for s in SCENARIOS]}")
    print(f"Step counts: {[steps_label(s) for s in STEP_COUNTS]}")
    print()

    # Launch all jobs in parallel
    futures = []
    for steps in STEP_COUNTS:
        for env_id, reward_path, label in SCENARIOS:
            futures.append(train_scenario.spawn(env_id, reward_path, label, steps))

    print(f"⏳ {len(futures)} jobs spawned. Waiting for completion...")
    start = time.time()

    # Collect results
    results = []
    for future in futures:
        result = future.get()
        results.append(result)
        slabel = steps_label(result["steps"])
        sr = result["metrics"].get("success_rate", "?")
        print(f"  ✅ {result['label']} @ {slabel}: success_rate={sr}")

    elapsed = time.time() - start
    print(f"\n🏁 All {len(results)} jobs complete in {elapsed:.0f}s")

    # Download metrics locally
    print("\nDownloading metrics...")
    local_project = str(project_root)
    for step_label in [steps_label(s) for s in STEP_COUNTS]:
        metrics = get_metrics.remote(step_label)
        if metrics:
            local_log_dir = os.path.join(local_project, "logs", f"sac_{step_label}")
            os.makedirs(local_log_dir, exist_ok=True)
            for label, m in metrics.items():
                path = os.path.join(local_log_dir, f"{label}_metrics.json")
                with open(path, "w") as f:
                    json.dump(m, f, indent=2)
                print(f"  Saved: {path}")

    print("\n✅ All metrics downloaded locally.")
    print("   Models in Modal volume: `modal volume get cs224r-sac-results models/ ./models/`")
