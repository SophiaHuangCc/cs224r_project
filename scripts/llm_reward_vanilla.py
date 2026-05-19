"""
Vanilla LLM Reward Generation — single-shot, no iteration.
Asks an LLM to write a reward function given task description + obs space.
"""
import os
import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Fetch envs
import numpy as np
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-12-01-preview",
)

TASK_PROMPT_TEMPLATE = """You are a reward function designer for robotic manipulation tasks.

Task: {task_name}
Task description: {task_desc}

Observation space:
{obs_space}

Action space: {action_space}

Write a Python reward function with this exact signature:
```python
def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # achieved_goal: np.array - current state achieved
    # desired_goal: np.array - target state
    # obs: dict with keys {obs_keys}
    # action: np.array of shape {action_shape}
    # info: dict with environment info
    # Return: float (scalar reward)
```

Design the reward to encourage the robot to complete the task efficiently.
Only output the function body, no imports (numpy is available as np).
"""


def get_task_config(env_id):
    """Get task-specific prompt info."""
    configs = {
        "FetchReach-v4": {
            "task_name": "FetchReach",
            "task_desc": "Move the robot gripper to a target 3D position in space.",
            "obs_keys": "observation, achieved_goal, desired_goal",
        },
        "FetchPickAndPlace-v4": {
            "task_name": "FetchPickAndPlace",
            "task_desc": "Pick up a block from the table and place it at a target 3D position (which may be in the air).",
            "obs_keys": "observation, achieved_goal, desired_goal",
        },
    }
    return configs[env_id]


# Default deployment name — override via AZURE_OPENAI_DEPLOYMENT env var
DEFAULT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def generate_reward_function(env_id: str, model: str = None) -> str:
    """Single-shot LLM reward generation."""
    env = gym.make(env_id)
    config = get_task_config(env_id)

    obs_space = env.observation_space
    action_space = env.action_space

    prompt = TASK_PROMPT_TEMPLATE.format(
        task_name=config["task_name"],
        task_desc=config["task_desc"],
        obs_space=str(obs_space),
        action_space=str(action_space),
        obs_keys=config["obs_keys"],
        action_shape=action_space.shape,
    )

    response = client.chat.completions.create(
        model=model or DEFAULT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    reward_code = response.choices[0].message.content
    # Strip markdown code fences if present
    reward_code = reward_code.strip()
    if reward_code.startswith("```"):
        reward_code = "\n".join(reward_code.split("\n")[1:])
    if reward_code.endswith("```"):
        reward_code = "\n".join(reward_code.split("\n")[:-1])

    env.close()
    return reward_code


def compile_reward_fn(code_str: str):
    """Compile LLM-generated code into a callable."""
    # Wrap in function definition if not already
    if not code_str.strip().startswith("def compute_reward"):
        code_str = (
            f"def compute_reward(achieved_goal, desired_goal, obs, action, info):\n"
            + "\n".join(f"    {line}" for line in code_str.split("\n"))
        )

    namespace = {"np": np}
    exec(code_str, namespace)
    return namespace["compute_reward"]


class VanillaLLMRewardWrapper(gym.Wrapper):
    """Replaces environment reward with LLM-generated reward."""

    def __init__(self, env, reward_fn, log_original=True):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.log_original = log_original
        self._last_action = None

    def step(self, action):
        self._last_action = action
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        try:
            llm_reward = self.reward_fn(
                obs["achieved_goal"],
                obs["desired_goal"],
                obs,
                action,
                info,
            )
        except Exception as e:
            # Fallback to original if LLM reward crashes
            print(f"LLM reward error: {e}, using original")
            llm_reward = original_reward

        if self.log_original:
            info["original_reward"] = original_reward
            info["llm_reward"] = llm_reward

        return obs, float(llm_reward), terminated, truncated, info


# --- Usage ---
if __name__ == "__main__":
    env_id = "FetchReach-v4"

    print(f"Generating reward for {env_id}...")
    reward_code = generate_reward_function(env_id)
    print(f"Generated reward:\n{reward_code}\n")

    # Save generated reward for reproducibility
    os.makedirs("generated_rewards", exist_ok=True)
    with open(f"generated_rewards/{env_id}_vanilla.py", "w") as f:
        f.write(reward_code)

    # Compile and wrap
    reward_fn = compile_reward_fn(reward_code)
    env = gym.make(env_id)
    env = VanillaLLMRewardWrapper(env, reward_fn)

    # Quick sanity check
    obs, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"LLM reward: {reward:.4f} | Original: {info['original_reward']:.4f}")
        if term or trunc:
            obs, _ = env.reset()
