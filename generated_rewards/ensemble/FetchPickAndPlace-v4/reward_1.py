def compute_reward(achieved_goal, desired_goal, obs, action, info):
    import numpy as np

    # Distance to the desired goal
    distance = np.linalg.norm(achieved_goal - desired_goal)

    # Reward shaping for intermediate milestones
    table_height = 0.5  # Assuming table height is known
    block_height = obs[3]  # Assuming block height is in observation[3]
    success_threshold = 0.05  # Threshold for considering the block at the target position

    # Success bonus for achieving the goal
    success_bonus = 10.0 if distance < success_threshold else 0.0

    # Reward shaping based on block being lifted
    lifted_bonus = 2.0 if block_height > table_height + 0.05 else 0.0

    # Penalize excessive actions
    action_penalty = -0.1 * np.linalg.norm(action)

    # Combine rewards
    reward = -distance + success_bonus + lifted_bonus + action_penalty
    return reward