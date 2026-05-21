"""
Eureka-style Iterative Reward Design (Pure — no safety mitigation).
Loop: LLM proposes reward → short training → evaluate task performance → feed metrics back → refine.
Based on: "Eureka: Human-Level Reward Design via Coding Large Language Models" (Ma et al. 2023)

NOTE: Safety metrics are LOGGED but NOT fed back to the LLM.
This isolates Eureka's contribution to task performance only.
"""
import os
import json
import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Fetch envs
import numpy as np
from stable_baselines3 import PPO
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

from llm_reward_vanilla import compile_reward_fn, VanillaLLMRewardWrapper, get_task_config

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-12-01-preview",
)

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


class SafetyMetricWrapper(gym.Wrapper):
    """Tracks safety-relevant metrics during episodes (for logging only)."""

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

        if terminated or truncated:
            info["safety_metrics"] = {
                "mean_action_mag": float(np.mean(self._episode_metrics["action_magnitudes"])),
                "mean_jerk": float(np.mean(self._episode_metrics["jerks"])) if self._episode_metrics["jerks"] else 0.0,
                "max_action_mag": float(self._episode_metrics["max_action_mag"]),
                "ep_length": self._episode_metrics["steps"],
            }

        return obs, reward, terminated, truncated, info


def evaluate_with_safety(model, env, n_episodes=50):
    """Evaluate policy — collect both task and safety metrics."""
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
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_action_mag": float(np.mean(action_mags)) if action_mags else 0,
        "mean_jerk": float(np.mean(jerks)) if jerks else 0,
        "max_action_mag": float(np.max(max_mags)) if max_mags else 0,
        "mean_ep_len": float(np.mean(ep_lens)) if ep_lens else 0,
    }


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
    Only task performance is fed back to the LLM. Safety metrics are logged but hidden from LLM.
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

        # Train with LLM reward
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

        # Evaluate on original env reward (to check true success) with safety logging
        eval_env = SafetyMetricWrapper(gym.make(env_id))
        metrics = evaluate_with_safety(model, eval_env, n_episodes=eval_episodes)

        print(f"Results: {json.dumps(metrics, indent=2)}")
        history.append(
            {"iteration": iteration + 1, "reward_code": reward_code, "metrics": metrics}
        )

        # Save intermediate results
        os.makedirs("generated_rewards/eureka", exist_ok=True)
        with open(f"generated_rewards/eureka/{env_id}_iter{iteration+1}.py", "w") as f:
            f.write(reward_code)
        with open(f"generated_rewards/eureka/{env_id}_iter{iteration+1}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save model checkpoint
        os.makedirs("models/eureka", exist_ok=True)
        model.save(f"models/eureka/{env_id}_iter{iteration+1}")

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
        eval_env.close()

    # Save full history
    with open(f"generated_rewards/eureka/{env_id}_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

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
