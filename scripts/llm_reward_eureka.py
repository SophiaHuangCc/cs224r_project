"""
Eureka-style Iterative Reward Design.
Loop: LLM proposes reward → short training → evaluate → feed metrics back → refine.
Based on: "Eureka: Human-Level Reward Design via Coding Large Language Models" (Ma et al. 2023)
"""
import os
import json
import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Fetch envs
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from openai import AzureOpenAI

from llm_reward_vanilla import compile_reward_fn, VanillaLLMRewardWrapper, get_task_config

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-12-01-preview",
)

DEFAULT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

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

Design a reward that helps the robot learn the task efficiently.
Think carefully about:
- Distance-based shaping
- Action penalties for smoothness
- Success bonuses
- Any intermediate milestones

Output ONLY the complete function (with def line). No explanation.
"""

REFINEMENT_PROMPT = """You previously designed this reward function:

```python
{previous_reward}
```

After training for {timesteps} steps, here are the evaluation results over {n_episodes} episodes:

- Success rate: {success_rate:.1%}
- Mean episode reward: {mean_reward:.2f}
- Mean action magnitude: {mean_action_mag:.4f}
- Mean jerk (action smoothness): {mean_jerk:.4f}
- Max action magnitude: {max_action_mag:.4f}
- Mean episode length: {mean_ep_len:.1f}

{analysis}

Please revise the reward function to improve performance. Consider:
1. If success rate is low, make the reward shaping stronger
2. If actions are too aggressive (high magnitude/jerk), add penalties
3. If episodes are too long, add time penalties or better shaping

Output ONLY the revised complete function. No explanation.
"""


def analyze_metrics(success_rate, mean_action_mag, mean_jerk):
    """Generate analysis text for the refinement prompt."""
    issues = []
    if success_rate < 0.5:
        issues.append(
            "Success rate is poor — the reward shaping may be too sparse or misleading."
        )
    if mean_action_mag > 0.3:
        issues.append(
            "Actions are very aggressive — the robot is using excessive force/velocity."
        )
    if mean_jerk > 0.15:
        issues.append(
            "Motion is jerky/non-smooth — consider penalizing action differences."
        )
    if not issues:
        issues.append(
            "Performance looks reasonable. Try to maintain success while reducing action aggressiveness."
        )
    return "\n".join(f"- {i}" for i in issues)


class SafetyMetricWrapper(gym.Wrapper):
    """Tracks safety-relevant metrics during episodes."""

    def __init__(self, env):
        super().__init__(env)
        self._prev_action = None
        self._episode_metrics = self._reset_metrics()

    def _reset_metrics(self):
        return {
            "action_magnitudes": [],
            "jerks": [],
            "max_action_mag": 0.0,
            "steps": 0,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = None
        self._episode_metrics = self._reset_metrics()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        mag = np.linalg.norm(action)
        self._episode_metrics["action_magnitudes"].append(mag)
        self._episode_metrics["max_action_mag"] = max(
            self._episode_metrics["max_action_mag"], mag
        )

        if self._prev_action is not None:
            jerk = np.linalg.norm(action - self._prev_action)
            self._episode_metrics["jerks"].append(jerk)

        self._prev_action = action.copy()
        self._episode_metrics["steps"] += 1

        # Attach metrics to info on episode end
        if terminated or truncated:
            info["safety_metrics"] = {
                "mean_action_mag": np.mean(
                    self._episode_metrics["action_magnitudes"]
                ),
                "mean_jerk": np.mean(self._episode_metrics["jerks"])
                if self._episode_metrics["jerks"]
                else 0.0,
                "max_action_mag": self._episode_metrics["max_action_mag"],
                "ep_length": self._episode_metrics["steps"],
            }

        return obs, reward, terminated, truncated, info


def evaluate_with_safety(model, env, n_episodes=50):
    """Evaluate policy and collect safety metrics."""
    successes = []
    rewards = []
    action_mags = []
    jerks = []
    max_mags = []
    ep_lens = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)
        successes.append(info.get("is_success", False))

        if "safety_metrics" in info:
            sm = info["safety_metrics"]
            action_mags.append(sm["mean_action_mag"])
            jerks.append(sm["mean_jerk"])
            max_mags.append(sm["max_action_mag"])
            ep_lens.append(sm["ep_length"])

    return {
        "success_rate": np.mean(successes),
        "mean_reward": np.mean(rewards),
        "mean_action_mag": np.mean(action_mags) if action_mags else 0,
        "mean_jerk": np.mean(jerks) if jerks else 0,
        "max_action_mag": np.max(max_mags) if max_mags else 0,
        "mean_ep_len": np.mean(ep_lens) if ep_lens else 0,
    }


def eureka_loop(
    env_id: str,
    n_iterations: int = 3,
    timesteps_per_iter: int = 30_000,
    eval_episodes: int = 50,
    model_name: str = None,
    seed: int = 42,
):
    """
    Eureka-style iterative reward refinement.

    Returns: list of (reward_code, metrics) per iteration
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

    response = client.chat.completions.create(
        model=model_name or DEFAULT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    reward_code = response.choices[0].message.content.strip()
    if reward_code.startswith("```"):
        reward_code = "\n".join(reward_code.split("\n")[1:])
    if reward_code.endswith("```"):
        reward_code = "\n".join(reward_code.split("\n")[:-1])

    env.close()

    history = []

    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"EUREKA ITERATION {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")
        print(f"Reward function:\n{reward_code}\n")

        # Compile reward
        try:
            reward_fn = compile_reward_fn(reward_code)
        except Exception as e:
            print(f"Compilation failed: {e}. Requesting fix...")
            # Ask LLM to fix syntax
            fix_response = client.chat.completions.create(
                model=model_name or DEFAULT_DEPLOYMENT,
                messages=[
                    {
                        "role": "user",
                        "content": f"This reward function has a syntax error:\n```python\n{reward_code}\n```\nError: {e}\nFix it. Output ONLY the corrected function.",
                    }
                ],
                temperature=0.3,
            )
            reward_code = fix_response.choices[0].message.content.strip()
            if reward_code.startswith("```"):
                reward_code = "\n".join(reward_code.split("\n")[1:])
            if reward_code.endswith("```"):
                reward_code = "\n".join(reward_code.split("\n")[:-1])
            reward_fn = compile_reward_fn(reward_code)

        # Train with this reward
        train_env = SafetyMetricWrapper(
            VanillaLLMRewardWrapper(gym.make(env_id), reward_fn)
        )

        model = PPO(
            "MultiInputPolicy",
            train_env,
            verbose=0,
            seed=seed + iteration,
            n_steps=1024,
            batch_size=64,
            learning_rate=3e-4,
        )
        model.learn(total_timesteps=timesteps_per_iter)

        # Evaluate with safety metrics on original env
        eval_wrapped = SafetyMetricWrapper(gym.make(env_id))
        metrics = evaluate_with_safety(model, eval_wrapped, n_episodes=eval_episodes)

        print(f"Results: {json.dumps(metrics, indent=2)}")
        history.append(
            {"iteration": iteration + 1, "reward_code": reward_code, "metrics": metrics}
        )

        # Save intermediate
        os.makedirs("generated_rewards/eureka", exist_ok=True)
        with open(
            f"generated_rewards/eureka/{env_id}_iter{iteration+1}.py", "w"
        ) as f:
            f.write(reward_code)
        with open(
            f"generated_rewards/eureka/{env_id}_iter{iteration+1}_metrics.json", "w"
        ) as f:
            json.dump(metrics, f, indent=2)

        # Refinement (skip on last iteration)
        if iteration < n_iterations - 1:
            analysis = analyze_metrics(
                metrics["success_rate"],
                metrics["mean_action_mag"],
                metrics["mean_jerk"],
            )

            refine_prompt = REFINEMENT_PROMPT.format(
                previous_reward=reward_code,
                timesteps=timesteps_per_iter,
                n_episodes=eval_episodes,
                analysis=analysis,
                **metrics,
            )

            response = client.chat.completions.create(
                model=model_name or DEFAULT_DEPLOYMENT,
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.7,
            )
            reward_code = response.choices[0].message.content.strip()
            if reward_code.startswith("```"):
                reward_code = "\n".join(reward_code.split("\n")[1:])
            if reward_code.endswith("```"):
                reward_code = "\n".join(reward_code.split("\n")[:-1])

        train_env.close()
        eval_wrapped.close()

    return history


if __name__ == "__main__":
    env_id = "FetchReach-v4"
    print(f"Running Eureka loop for {env_id}...")

    history = eureka_loop(
        env_id=env_id,
        n_iterations=3,
        timesteps_per_iter=30_000,
        eval_episodes=50,
    )

    # Summary
    print("\n\n" + "=" * 60)
    print("EUREKA SUMMARY")
    print("=" * 60)
    for h in history:
        m = h["metrics"]
        print(
            f"Iter {h['iteration']}: success={m['success_rate']:.1%} | "
            f"action_mag={m['mean_action_mag']:.3f} | "
            f"jerk={m['mean_jerk']:.3f}"
        )
