"""
Mitigation 1: Reward Ensemble.

Generate N independent Eureka-quality reward functions and aggregate them into
a single reward signal. A loophole exploitable in one reward is unlikely to
exist across all N.

Aggregation strategies:
- "min" (conservative): R = min(r_1, ..., r_N) — strongest anti-hacking
- "mean":               R = mean(r_1, ..., r_N) — smooths quirks
- "trimmed_mean":       drop highest & lowest, average the rest (N >= 3)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable, List, Literal, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_reward_eureka_sac import eureka_loop_sac  # noqa: E402


Aggregation = Literal["min", "mean", "trimmed_mean"]


def generate_eureka_ensemble_rewards(
    env_id: str,
    n: int = 3,
    eureka_iters: int = 3,
    timesteps_per_iter: int = 50_000,
    eval_episodes: int = 50,
    save_dir: Optional[str] = None,
    seed: int = 42,
) -> List[str]:
    """Run N independent Eureka loops (SAC+HER), return the best reward from each.

    Each loop independently generates + iteratively refines a reward function
    using SAC+HER for evaluation, so the N best rewards come from N different
    optimization trajectories.

    Returns a list of N reward code strings (the best from each Eureka run).
    """
    reward_codes: List[str] = []

    for i in range(n):
        print(f"\n{'='*60}")
        print(f"  EUREKA ENSEMBLE: independent loop {i+1}/{n} (SAC+HER)")
        print(f"  Env: {env_id} | Iters: {eureka_iters} | Steps/iter: {timesteps_per_iter}")
        print(f"{'='*60}")

        history = eureka_loop_sac(
            env_id=env_id,
            n_iterations=eureka_iters,
            timesteps_per_iter=timesteps_per_iter,
            eval_episodes=eval_episodes,
            seed=seed + i * 100,
        )

        best = max(history, key=lambda h: h["metrics"]["success_rate"])
        print(
            f"  Loop {i+1} best: iter {best['iteration']} "
            f"(success={best['metrics']['success_rate']:.1%})"
        )
        reward_codes.append(best["reward_code"])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i, code in enumerate(reward_codes):
            with open(os.path.join(save_dir, f"reward_{i+1}.py"), "w") as f:
                f.write(code)
        meta = {
            "env_id": env_id,
            "n": n,
            "eureka_iters": eureka_iters,
            "timesteps_per_iter": timesteps_per_iter,
            "method": "independent_eureka_loops_sac_her",
        }
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  Saved {n} Eureka-quality rewards to {save_dir}/")

    return reward_codes


def make_ensemble_reward_fn(
    reward_fns: List[Callable],
    aggregation: Aggregation = "min",
) -> Callable:
    """Combine N reward functions into one that returns the aggregate per call.

    Pass the returned callable to `evaluate_with_safety(reward_fn=...)` so
    hacking_rate is computed against the ensemble (not individual functions).
    """
    def ensemble_fn(achieved_goal, desired_goal, obs, action, info):
        individual = []
        for fn in reward_fns:
            try:
                individual.append(float(fn(achieved_goal, desired_goal, obs, action, info)))
            except Exception:
                individual.append(0.0)
        arr = np.array(individual, dtype=np.float64)
        if aggregation == "min":
            return float(np.min(arr))
        if aggregation == "mean":
            return float(np.mean(arr))
        if aggregation == "trimmed_mean":
            if len(arr) < 3:
                return float(np.mean(arr))
            return float(np.mean(np.sort(arr)[1:-1]))
        return float(np.mean(arr))
    return ensemble_fn
