"""
Eureka-style Iterative Reward Design (Pure — no safety mitigation).

Loop: LLM proposes reward → short PPO training (delegated to `ppo.training`)
→ evaluate task performance → feed metrics back → refine.

Based on: "Eureka: Human-Level Reward Design via Coding Large Language Models"
(Ma et al. 2023).

This file only handles **reward generation / refinement**. PPO training lives
in `scripts/ppo/`, safety-metric tracking + evaluation lives in
`scripts/safety/`. The output of running this script is a set of generated
reward files under `generated_rewards/eureka_ppo/`. A separate PPO training script
can read those reward files and run full-length training without re-invoking
the LLM.

NOTE: Safety metrics are LOGGED but NOT fed back to the LLM.
This isolates Eureka's contribution to task performance only.
"""
import os
import sys
import json

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Fetch envs
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Make sibling modules importable when invoked from repo root or scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_reward_vanilla import compile_reward_fn, get_task_config  # noqa: E402
from ppo import train_ppo_with_reward  # noqa: E402

# Lazily instantiated so importing this module doesn't require Azure creds.
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-12-01-preview",
        )
    return _client


DEFAULT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# --- Prompts (performance-only, no safety hints) ---

INITIAL_PROMPT = """You are an expert reward function designer for robotic manipulation.

Task: {task_name}
Task description: {task_desc}
Observation space: {obs_space}
Action space: {action_space}

Write a reward function with this signature:
```python
def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Return: float
```

Design a reward that helps the robot learn the task as efficiently as possible.
Think about distance-based shaping, success bonuses, and intermediate milestones.

Output ONLY the complete function (with def line). No explanation.
"""

REFINEMENT_PROMPT = """You previously designed this reward function:

```python
{previous_reward}
```

After training for {timesteps} steps, here are the results over {n_episodes} evaluation episodes:

- Success rate: {success_rate:.1%}
- Mean episode reward: {mean_reward:.2f}
- Mean episode length: {mean_ep_len:.1f} steps

{analysis}

Revise the reward function to improve task success rate and learning efficiency.
Output ONLY the revised complete function. No explanation.
"""


def analyze_performance(success_rate, mean_ep_len):
    """Generate performance-only analysis (no safety feedback)."""
    issues = []
    if success_rate < 0.3:
        issues.append("Success rate is very low — reward shaping may be too sparse or misaligned with the actual goal.")
    elif success_rate < 0.7:
        issues.append("Success rate is moderate — reward shaping could be stronger or better aligned.")
    else:
        issues.append("Success rate is good. Try to make learning even more sample-efficient.")

    if mean_ep_len > 40:
        issues.append("Episodes are long — the agent may be wandering. Consider stronger goal-directed shaping.")

    return "\n".join(f"- {i}" for i in issues)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    return text


def eureka_loop(
    env_id: str,
    n_iterations: int = 3,
    timesteps_per_iter: int = 50_000,
    eval_episodes: int = 50,
    model_name: str = None,
    seed: int = 42,
):
    """
    Pure Eureka iterative reward refinement.

    Reward generation lives here; PPO training+evaluation is delegated to
    `ppo.train_ppo_with_reward` (which uses `safety.SafetyMetricWrapper` and
    `safety.evaluate_with_safety` under the hood). Each iteration's reward
    code and metrics are saved to `generated_rewards/eureka_ppo/`. A separate PPO
    training script can later read those rewards and run full-length training.

    Only task performance is fed back to the LLM. Safety metrics are logged
    but hidden from the LLM.
    """
    env = gym.make(env_id)
    config = get_task_config(env_id)

    # --- Initial reward generation ---
    prompt = INITIAL_PROMPT.format(
        task_name=config["task_name"],
        task_desc=config["task_desc"],
        obs_space=str(env.observation_space),
        action_space=str(env.action_space),
    )
    env.close()

    response = _get_client().chat.completions.create(
        model=model_name or DEFAULT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    reward_code = _strip_code_fences(response.choices[0].message.content)

    history = []
    out_dir = "generated_rewards/eureka_ppo"
    os.makedirs(out_dir, exist_ok=True)

    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"EUREKA ITERATION {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")
        print(f"Reward function:\n{reward_code}\n")

        # Compile reward (with one auto-fix retry on syntax errors)
        try:
            reward_fn = compile_reward_fn(reward_code)
        except Exception as e:
            print(f"Compilation failed: {e}. Requesting fix...")
            fix_response = _get_client().chat.completions.create(
                model=model_name or DEFAULT_DEPLOYMENT,
                messages=[
                    {
                        "role": "user",
                        "content": f"This reward function has a syntax error:\n```python\n{reward_code}\n```\nError: {e}\nFix it. Output ONLY the corrected function.",
                    }
                ],
                temperature=0.3,
            )
            reward_code = _strip_code_fences(fix_response.choices[0].message.content)
            reward_fn = compile_reward_fn(reward_code)

        # Train + evaluate (delegated to the PPO module, which uses safety.*)
        _, metrics = train_ppo_with_reward(
            env_id=env_id,
            reward_fn=reward_fn,
            timesteps=timesteps_per_iter,
            seed=seed + iteration,
            eval_episodes=eval_episodes,
            save_path=f"models/eureka/{env_id}_iter{iteration+1}",
        )

        print(f"Results: {json.dumps(metrics, indent=2)}")
        history.append(
            {"iteration": iteration + 1, "reward_code": reward_code, "metrics": metrics}
        )

        # Persist per-iteration reward + metrics
        with open(f"{out_dir}/{env_id}_iter{iteration+1}.py", "w") as f:
            f.write(reward_code)
        with open(f"{out_dir}/{env_id}_iter{iteration+1}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # --- Refinement: ONLY task performance fed back ---
        if iteration < n_iterations - 1:
            analysis = analyze_performance(
                metrics["success_rate"],
                metrics["mean_ep_len"],
            )

            refine_prompt = REFINEMENT_PROMPT.format(
                previous_reward=reward_code,
                timesteps=timesteps_per_iter,
                n_episodes=eval_episodes,
                success_rate=metrics["success_rate"],
                mean_reward=metrics["mean_reward"],
                mean_ep_len=metrics["mean_ep_len"],
                analysis=analysis,
            )

            response = _get_client().chat.completions.create(
                model=model_name or DEFAULT_DEPLOYMENT,
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.7,
            )
            reward_code = _strip_code_fences(response.choices[0].message.content)

    # Save full history
    with open(f"{out_dir}/{env_id}_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Save best reward as `<env_id>_best.py` for downstream PPO training
    best = max(history, key=lambda h: h["metrics"]["success_rate"])
    with open(f"{out_dir}/{env_id}_best.py", "w") as f:
        f.write(best["reward_code"])
    with open(f"{out_dir}/{env_id}_best_metrics.json", "w") as f:
        json.dump(best["metrics"], f, indent=2)

    return history


if __name__ == "__main__":
    env_id = "FetchReach-v4"
    print(f"Running Pure Eureka loop for {env_id}...")

    history = eureka_loop(
        env_id=env_id,
        n_iterations=3,
        timesteps_per_iter=50_000,
        eval_episodes=50,
    )

    # Summary
    print("\n\n" + "=" * 60)
    print("EUREKA SUMMARY (Pure — no safety feedback to LLM)")
    print("=" * 60)
    for h in history:
        m = h["metrics"]
        print(
            f"Iter {h['iteration']}: success={m['success_rate']:.1%} | "
            f"reward={m['mean_reward']:.2f} | "
            f"action_mag={m['mean_action_mag']:.3f} | "
            f"jerk={m['mean_jerk']:.3f} | "
            f"ep_len={m['mean_ep_len']:.0f}"
        )
