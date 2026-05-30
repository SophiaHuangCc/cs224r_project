"""
Mitigation 1: Reward Ensemble

Generate N independent LLM reward functions (varied temperature) and aggregate
them. A loophole exploitable in one reward is unlikely to exist across all N.

Aggregation strategies:
- "min" (conservative): R = min(r_1, ..., r_N) — strongest anti-hacking
- "mean": R = mean(r_1, ..., r_N) — smooths quirks
- "trimmed_mean": drop highest & lowest, average the rest (N>=3)
"""
from __future__ import annotations

import os
import sys
import json
from typing import Callable, List, Literal, Optional

import gymnasium as gym
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_reward_vanilla import compile_reward_fn, get_task_config, VanillaLLMRewardWrapper
from llm_reward_eureka_sac import eureka_loop_sac
from sac.training import HERLLMRewardWrapper, train_sac_her_with_reward


# --------------------------------------------------------------------------- #
# Azure OpenAI client                                                          #
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Reward generation prompt (same as eureka initial, varied per call)           #
# --------------------------------------------------------------------------- #

ENSEMBLE_PROMPT_VARIANTS = [
    # Variant 1: standard
    """You are an expert reward function designer for robotic manipulation.

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

Output ONLY the complete function (with def line). No explanation.""",

    # Variant 2: emphasize different approach
    """You are a robotics reward engineer. Design a reward function for:

Task: {task_name}
Description: {task_desc}
Observation space: {obs_space}
Action space: {action_space}

Signature:
```python
def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Return: float
```

Focus on creating smooth, well-shaped gradients that guide the robot toward task completion.
Consider what intermediate behaviors should be encouraged.

Output ONLY the complete Python function. No explanation.""",

    # Variant 3: another perspective
    """As a reward shaping specialist, write a reward function for a robot manipulation task.

Task: {task_name}
What the robot should do: {task_desc}
Observation space: {obs_space}
Action space: {action_space}

Function signature:
```python
def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Return: float
```

Think carefully about what progress looks like for this task and reward it proportionally.
Avoid reward functions that can be trivially exploited without completing the task.

Output ONLY the function code. No explanation or markdown outside the function.""",

    # Variant 4: concise, different framing
    """Design a dense reward function for robotic {task_name}.

Task: {task_desc}
Obs: {obs_space}
Actions: {action_space}

```python
def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Return: float
```

Reward should incentivize task completion with shaped intermediate signals.
Output the function only.""",

    # Variant 5: emphasis on robustness
    """You are designing a reward function that will be used to train a robot via reinforcement learning.

Task: {task_name} — {task_desc}
Observation: {obs_space}
Action: {action_space}

Write:
```python
def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Return: float
```

The reward should be robust — it should only give high reward when the task is genuinely being completed, not for degenerate behaviors.

Output ONLY the function. No explanation.""",
]


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    return text


# --------------------------------------------------------------------------- #
# Generate N independent reward functions (vanilla single-shot)                #
# --------------------------------------------------------------------------- #
def generate_ensemble_rewards(
    env_id: str,
    n: int = 3,
    temperatures: Optional[List[float]] = None,
    model: str = None,
    save_dir: Optional[str] = None,
) -> List[str]:
    """Generate N independent LLM reward functions using varied prompts + temperatures.

    Returns list of N reward code strings. (Single-shot, no iteration.)
    """
    if temperatures is None:
        temperatures = [0.5, 0.7, 0.9, 1.1, 1.3][:n]
    assert len(temperatures) >= n, f"Need at least {n} temperatures, got {len(temperatures)}"

    env = gym.make(env_id)
    config = get_task_config(env_id)
    obs_space = str(env.observation_space)
    action_space = str(env.action_space)
    env.close()

    reward_codes = []
    for i in range(n):
        # Cycle through prompt variants
        prompt_template = ENSEMBLE_PROMPT_VARIANTS[i % len(ENSEMBLE_PROMPT_VARIANTS)]
        prompt = prompt_template.format(
            task_name=config["task_name"],
            task_desc=config["task_desc"],
            obs_space=obs_space,
            action_space=action_space,
        )

        temp = temperatures[i]
        print(f"  Generating reward {i+1}/{n} (temperature={temp})...")

        response = _get_client().chat.completions.create(
            model=model or DEFAULT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
        )
        code = _strip_code_fences(response.choices[0].message.content)

        # Validate it compiles
        try:
            compile_reward_fn(code)
        except Exception as e:
            print(f"  ⚠️  Reward {i+1} failed to compile: {e}. Requesting fix...")
            fix_response = _get_client().chat.completions.create(
                model=model or DEFAULT_DEPLOYMENT,
                messages=[{
                    "role": "user",
                    "content": f"This reward function has an error:\n```python\n{code}\n```\nError: {e}\nFix it. Output ONLY the corrected function.",
                }],
                temperature=0.3,
            )
            code = _strip_code_fences(fix_response.choices[0].message.content)
            compile_reward_fn(code)  # will raise if still broken

        reward_codes.append(code)

    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i, code in enumerate(reward_codes):
            with open(os.path.join(save_dir, f"reward_{i+1}.py"), "w") as f:
                f.write(code)
        print(f"  Saved {n} rewards to {save_dir}/")

    return reward_codes


# --------------------------------------------------------------------------- #
# Generate N independent EUREKA-quality reward functions (SAC+HER)              #
# --------------------------------------------------------------------------- #
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
    using SAC+HER for evaluation (same algorithm as final training), so the
    N best rewards come from N different optimization trajectories.

    Returns list of N reward code strings (the best from each Eureka run).
    """
    reward_codes = []

    for i in range(n):
        print(f"\n{'='*60}")
        print(f"  EUREKA ENSEMBLE: Running independent loop {i+1}/{n} (SAC+HER)")
        print(f"  Env: {env_id} | Iters: {eureka_iters} | Steps/iter: {timesteps_per_iter}")
        print(f"{'='*60}")

        loop_seed = seed + i * 100

        history = eureka_loop_sac(
            env_id=env_id,
            n_iterations=eureka_iters,
            timesteps_per_iter=timesteps_per_iter,
            eval_episodes=eval_episodes,
            seed=loop_seed,
        )

        best = max(history, key=lambda h: h["metrics"]["success_rate"])
        best_code = best["reward_code"]
        best_sr = best["metrics"]["success_rate"]

        print(f"  Loop {i+1} best: iter {best['iteration']} (success={best_sr:.1%})")
        reward_codes.append(best_code)

    # Save
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


# --------------------------------------------------------------------------- #
# Ensemble Reward Wrapper                                                      #
# --------------------------------------------------------------------------- #
class EnsembleRewardWrapper(gym.Wrapper):
    """Aggregates N reward functions into a single reward signal.

    Works with HER by exposing `compute_reward(achieved, desired, info)`.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_fns: List[Callable],
        aggregation: Literal["min", "mean", "trimmed_mean"] = "min",
        log_individual: bool = True,
    ):
        super().__init__(env)
        self.reward_fns = reward_fns
        self.aggregation = aggregation
        self.log_individual = log_individual
        self._last_action = None

    def _aggregate(self, rewards: List[float]) -> float:
        arr = np.array(rewards, dtype=np.float64)
        if self.aggregation == "min":
            return float(np.min(arr))
        elif self.aggregation == "mean":
            return float(np.mean(arr))
        elif self.aggregation == "trimmed_mean":
            if len(arr) < 3:
                return float(np.mean(arr))
            sorted_arr = np.sort(arr)
            return float(np.mean(sorted_arr[1:-1]))
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def step(self, action):
        self._last_action = action
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        individual_rewards = []
        for i, fn in enumerate(self.reward_fns):
            try:
                r = float(fn(
                    obs["achieved_goal"],
                    obs["desired_goal"],
                    obs,
                    action,
                    info,
                ))
            except Exception as e:
                # Fallback: use original env reward for this function
                r = float(original_reward)
                if self.log_individual:
                    print(f"Ensemble reward {i} error: {e}, using original")
            individual_rewards.append(r)

        ensemble_reward = self._aggregate(individual_rewards)

        if self.log_individual:
            info["original_reward"] = original_reward
            info["ensemble_rewards"] = individual_rewards
            info["ensemble_aggregated"] = ensemble_reward

        return obs, ensemble_reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """HER-compatible batched/unbatched compute_reward."""
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)

        if achieved_goal.ndim >= 2:
            batch_size = achieved_goal.shape[0]
            if isinstance(info, (list, tuple, np.ndarray)) and len(info) == batch_size:
                infos = list(info)
            else:
                infos = [info] * batch_size

            rewards = np.empty(batch_size, dtype=np.float32)
            for i in range(batch_size):
                individual = []
                for fn in self.reward_fns:
                    try:
                        individual.append(float(fn(
                            achieved_goal[i], desired_goal[i], None, None, infos[i]
                        )))
                    except Exception:
                        # Fallback to env reward for this function
                        individual.append(float(
                            self.env.unwrapped.compute_reward(
                                achieved_goal[i], desired_goal[i], infos[i]
                            )
                        ))
                rewards[i] = self._aggregate(individual)
            return rewards

        # Unbatched
        individual = []
        for fn in self.reward_fns:
            try:
                individual.append(float(fn(achieved_goal, desired_goal, None, None, info)))
            except Exception:
                individual.append(float(
                    self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)
                ))
        return self._aggregate(individual)


# --------------------------------------------------------------------------- #
# Combined ensemble reward function (for evaluate_with_safety hacking_rate)    #
# --------------------------------------------------------------------------- #
def make_ensemble_reward_fn(
    reward_fns: List[Callable],
    aggregation: Literal["min", "mean", "trimmed_mean"] = "min",
) -> Callable:
    """Create a single callable that computes the ensemble reward.

    This is passed to `evaluate_with_safety(reward_fn=...)` so hacking_rate
    is computed against the ensemble reward (not individual functions).
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
        elif aggregation == "mean":
            return float(np.mean(arr))
        elif aggregation == "trimmed_mean":
            if len(arr) < 3:
                return float(np.mean(arr))
            sorted_arr = np.sort(arr)
            return float(np.mean(sorted_arr[1:-1]))
        return float(np.mean(arr))
    return ensemble_fn


# --------------------------------------------------------------------------- #
# CLI entry point                                                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ensemble reward functions")
    parser.add_argument("--env", default="FetchPickAndPlace-v4")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    save_dir = args.save_dir or f"generated_rewards/ensemble/{args.env}"
    print(f"Generating {args.n} reward functions for {args.env}...")
    codes = generate_ensemble_rewards(args.env, n=args.n, save_dir=save_dir)

    print(f"\n✅ Generated {len(codes)} rewards. Saved to {save_dir}/")
    for i, code in enumerate(codes):
        print(f"\n--- Reward {i+1} ---")
        print(code[:200] + "..." if len(code) > 200 else code)
