"""
Standalone CLI: train SAC+HER on a Fetch env using a reward function loaded
from `generated_rewards/`.

Supports checkpoint evaluation: train once, eval at multiple step counts.

Examples:
    # Basic training
    python scripts/sac/train_from_reward.py \
        --env FetchPickAndPlace-v4 \
        --reward generated_rewards/eureka_sac/FetchPickAndPlace-v4_best.py \
        --timesteps 500000

    # With checkpoints (eval at 100k, 250k, 500k)
    python scripts/sac/train_from_reward.py \
        --env FetchPickAndPlace-v4 \
        --reward generated_rewards/eureka_sac/FetchPickAndPlace-v4_best.py \
        --timesteps 500000 \
        --checkpoints 100000 250000 500000

    # Ensemble reward (multiple reward files aggregated)
    python scripts/sac/train_from_reward.py \
        --env FetchPickAndPlace-v4 \
        --reward generated_rewards/ensemble/FetchPickAndPlace-v4/reward_1.py \
                 generated_rewards/ensemble/FetchPickAndPlace-v4/reward_2.py \
                 generated_rewards/ensemble/FetchPickAndPlace-v4/reward_3.py \
        --aggregation min \
        --timesteps 500000 \
        --checkpoints 100000 250000 500000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make sibling packages importable when run as a script.
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS_DIR)

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
from stable_baselines3 import SAC, HerReplayBuffer

from sac import load_reward_fn_from_file, train_sac_her_with_reward  # noqa: E402
from sac.training import HERLLMRewardWrapper  # noqa: E402
from safety import SafetyMetricWrapper, evaluate_with_safety  # noqa: E402


# SAC+HER defaults (same everywhere for fair comparison)
SAC_KWARGS = dict(
    learning_rate=1e-3,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.95,
    tau=0.05,
    learning_starts=1_000,
)
HER_KWARGS = dict(
    n_sampled_goal=4,
    goal_selection_strategy="future",
)


def train_with_checkpoints(
    env_id: str,
    reward_fn,
    timesteps: int,
    checkpoints: list[int],
    seed: int = 42,
    eval_episodes: int = 100,
    save_dir: str = "models",
    results_dir: str = "results",
    label: str = "",
    tensorboard_log: str = None,
    verbose: int = 1,
) -> dict[int, dict]:
    """Train SAC+HER once, save checkpoints and eval at each step count.

    Returns dict of {checkpoint_steps: metrics}.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Ensure checkpoints are sorted and include final timesteps
    checkpoints = sorted(set(checkpoints))
    if timesteps not in checkpoints:
        checkpoints.append(timesteps)

    # Create training env
    train_env = SafetyMetricWrapper(
        HERLLMRewardWrapper(gym.make(env_id), reward_fn)
    )

    model = SAC(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=HER_KWARGS,
        verbose=verbose,
        seed=seed,
        tensorboard_log=tensorboard_log,
        **SAC_KWARGS,
    )

    all_metrics = {}
    steps_trained = 0
    start_time = time.time()

    for checkpoint in checkpoints:
        steps_to_train = checkpoint - steps_trained
        if steps_to_train <= 0:
            continue

        print(f"\n  Training {steps_trained//1000}k → {checkpoint//1000}k ({steps_to_train:,} steps)...")
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        steps_trained = checkpoint

        # Save model
        model_path = os.path.join(save_dir, f"{label}_{checkpoint//1000}k")
        model.save(model_path)
        print(f"  Model saved: {model_path}")

        # Evaluate
        print(f"  Evaluating ({eval_episodes} episodes)...")
        eval_env = SafetyMetricWrapper(gym.make(env_id))
        metrics = evaluate_with_safety(
            model, eval_env, n_episodes=eval_episodes, reward_fn=reward_fn
        )
        eval_env.close()

        metrics["checkpoint_steps"] = checkpoint
        results_path = os.path.join(results_dir, f"{label}_{checkpoint//1000}k.json")
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  {checkpoint//1000}k: success={metrics['success_rate']:.1%} | "
              f"hacking={metrics.get('hacking_rate', 'N/A')} | "
              f"action_mag={metrics['mean_action_mag']:.4f}")
        all_metrics[checkpoint] = metrics

    train_time = time.time() - start_time
    train_env.close()

    # Save summary
    summary = {
        "config": {
            "env_id": env_id,
            "label": label,
            "checkpoints": checkpoints,
            "eval_episodes": eval_episodes,
            "seed": seed,
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


def main():
    parser = argparse.ArgumentParser(description="Train SAC+HER from saved reward file(s).")
    parser.add_argument("--env", required=True, help="Gym env id, e.g. FetchPickAndPlace-v4")
    parser.add_argument("--reward", required=True, nargs="+",
                        help="Path(s) to reward .py file(s). Multiple = ensemble.")
    parser.add_argument("--aggregation", choices=["min", "mean", "trimmed_mean"], default="min",
                        help="Aggregation for ensemble (only used with multiple --reward files)")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=None,
                        help="Steps at which to checkpoint + eval (default: just final)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--label", default=None,
                        help="Label for output files (default: derived from reward path)")
    parser.add_argument("--tensorboard-log", default=None)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    # Load reward function(s)
    if len(args.reward) == 1:
        # Single reward
        reward_fn = load_reward_fn_from_file(args.reward[0])
        default_label = os.path.splitext(os.path.basename(args.reward[0]))[0]
    else:
        # Ensemble
        from ensemble_reward import make_ensemble_reward_fn

        reward_fns = [load_reward_fn_from_file(p) for p in args.reward]
        reward_fn = make_ensemble_reward_fn(reward_fns, aggregation=args.aggregation)
        default_label = f"{args.env}_ensemble_n{len(args.reward)}_{args.aggregation}"

    label = args.label or default_label
    save_dir = args.save_dir or f"models/sac_her/{label}"
    results_dir = args.results_dir or f"results/sac_her/{label}"

    print(f"\n{'='*60}")
    print(f"SAC+HER Training")
    print(f"  Env: {args.env}")
    print(f"  Reward: {args.reward}")
    if len(args.reward) > 1:
        print(f"  Ensemble: N={len(args.reward)}, aggregation={args.aggregation}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Label: {label}")
    print(f"{'='*60}")

    if args.checkpoints:
        # Train with checkpoints
        all_metrics = train_with_checkpoints(
            env_id=args.env,
            reward_fn=reward_fn,
            timesteps=args.timesteps,
            checkpoints=args.checkpoints,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            save_dir=save_dir,
            results_dir=results_dir,
            label=label,
            tensorboard_log=args.tensorboard_log,
            verbose=args.verbose,
        )

        # Print summary table
        print(f"\n{'='*60}")
        print(f"{'Checkpoint':<12} {'Success':>10} {'Hacking':>10} {'Action Mag':>12}")
        print(f"{'-'*46}")
        for ckpt in sorted(all_metrics.keys()):
            m = all_metrics[ckpt]
            hack_str = f"{m['hacking_rate']:.1%}" if 'hacking_rate' in m else "N/A"
            print(f"{ckpt//1000}k{'':<8} {m['success_rate']:.1%}{'':<5} "
                  f"{hack_str:<10} {m['mean_action_mag']:.4f}")
        print(f"{'='*60}")
    else:
        # Single training run (backward compatible)
        save_path = os.path.join(save_dir, label)
        _, metrics = train_sac_her_with_reward(
            env_id=args.env,
            reward_fn=reward_fn,
            timesteps=args.timesteps,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            save_path=save_path,
            tensorboard_log=args.tensorboard_log,
            verbose=args.verbose,
        )

        metrics_path = f"{save_path}_metrics.json"
        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nSaved model to {save_path}")
        print(f"Saved metrics to {metrics_path}")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
