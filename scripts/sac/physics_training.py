"""
SAC + HER training with a physics-constraint penalty wrapper (Mitigation 6).

Unlike `kl_training.py`, this needs no SAC subclass — the penalty lives in
the env wrapper, so the standard SAC optimizer is unchanged. We only swap
the env-construction step to insert `PhysicsConstraintRewardWrapper`
between `HERLLMRewardWrapper` and `SafetyMetricWrapper`. Placement matters:

    SafetyMetricWrapper                (outer — observes RAW behavior, untouched)
      PhysicsConstraintRewardWrapper   (modifies training reward only)
        HERLLMRewardWrapper            (LLM proxy reward + HER compute_reward)
          gym.make(env_id)

HER relabeling walks the wrapper chain to find `compute_reward`; since
`PhysicsConstraintRewardWrapper` does not override it, `gym.Wrapper.
__getattr__` forwards the lookup down to `HERLLMRewardWrapper.compute_reward`
as desired.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Fetch envs
from stable_baselines3 import HerReplayBuffer, SAC

from llm_reward_vanilla import VanillaLLMRewardWrapper  # noqa: F401
from safety import (
    PHYSICS_PRESETS,
    PhysicsConstraintRewardWrapper,
    SafetyMetricWrapper,
    evaluate_with_safety,
)

from .training import HER_KWARGS, HERLLMRewardWrapper, SAC_KWARGS, _make_model


def _make_physics_train_env(
    env_id: str,
    reward_fn: Callable,
    coef: float,
    thresholds: Optional[dict] = None,
) -> SafetyMetricWrapper:
    """Build training env with the physics-constraint penalty in the middle."""
    if thresholds is None:
        thresholds = PHYSICS_PRESETS.get(env_id, {})
    return SafetyMetricWrapper(
        PhysicsConstraintRewardWrapper(
            HERLLMRewardWrapper(gym.make(env_id), reward_fn),
            coef=coef,
            **thresholds,
        )
    )


def train_with_physics_checkpoints(
    env_id: str,
    reward_fn: Callable,
    coef: float,
    timesteps: int,
    checkpoints: list[int],
    thresholds: Optional[dict] = None,
    seed: int = 42,
    eval_episodes: int = 100,
    save_dir: str = "models/physics",
    results_dir: str = "results/physics",
    label: str = "",
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
) -> dict[int, dict]:
    """Train SAC+HER with physics-constraint penalty, evaluating at each checkpoint.

    Mirrors `train_with_checkpoints`. Evaluation uses a plain SafetyMetricWrapper
    env (no physics penalty) so the reported success / hacking / safety numbers
    reflect the true task, not the training-time modified reward.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    checkpoints = sorted(set(checkpoints))
    if timesteps not in checkpoints:
        checkpoints.append(timesteps)

    train_env = _make_physics_train_env(env_id, reward_fn, coef, thresholds)
    model = _make_model(train_env, seed=seed, tensorboard_log=tensorboard_log, verbose=verbose)

    all_metrics: dict[int, dict] = {}
    steps_trained = 0
    start_time = time.time()

    for checkpoint in checkpoints:
        steps_to_train = checkpoint - steps_trained
        if steps_to_train <= 0:
            continue

        print(f"\n  Training {steps_trained//1000}k → {checkpoint//1000}k ({steps_to_train:,} steps)...")
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        steps_trained = checkpoint

        model_path = os.path.join(save_dir, f"{label}_{checkpoint//1000}k")
        model.save(model_path)
        if os.path.exists(model_path) and not model_path.endswith(".zip"):
            os.rename(model_path, model_path + ".zip")
        print(f"  Saved: {model_path}.zip")

        print(f"  Evaluating ({eval_episodes} episodes)...")
        eval_env = SafetyMetricWrapper(gym.make(env_id))
        metrics = evaluate_with_safety(
            model, eval_env, n_episodes=eval_episodes, reward_fn=reward_fn
        )
        eval_env.close()

        metrics["checkpoint_steps"] = checkpoint
        metrics["physics_coef"] = coef
        results_path = os.path.join(results_dir, f"{label}_{checkpoint//1000}k.json")
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(
            f"  {checkpoint//1000}k: success={metrics['success_rate']:.1%} | "
            f"hacking={metrics.get('hacking_rate', 'N/A')} | "
            f"action_mag={metrics['mean_action_mag']:.4f}"
        )
        all_metrics[checkpoint] = metrics

    train_time = time.time() - start_time
    train_env.close()

    summary = {
        "config": {
            "env_id": env_id,
            "label": label,
            "checkpoints": checkpoints,
            "eval_episodes": eval_episodes,
            "seed": seed,
            "physics_coef": coef,
            "thresholds": thresholds if thresholds is not None else PHYSICS_PRESETS.get(env_id, {}),
            "train_time_seconds": train_time,
        },
        "results_by_checkpoint": {
            f"{c//1000}k": all_metrics[c] for c in checkpoints if c in all_metrics
        },
    }
    summary_path = os.path.join(results_dir, f"{label}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total train time: {train_time/60:.1f} min")
    print(f"  Summary: {summary_path}")

    return all_metrics
