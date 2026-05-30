def compute_reward(achieved_goal, desired_goal, obs, action, info):
    distance = np.linalg.norm(achieved_goal - desired_goal)
    success_threshold = 0.05
    reward_scale = 20.0  # Increased scale for shaping rewards
    success_bonus = 100.0  # Stronger incentive for success
    if distance < success_threshold:
        return float(success_bonus)  # Higher reward for achieving the goal
    shaping_reward = -reward_scale * distance
    return float(shaping_reward)  # Scaled shaping reward to encourage progress