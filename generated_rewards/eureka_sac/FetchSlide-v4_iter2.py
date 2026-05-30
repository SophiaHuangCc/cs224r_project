def compute_reward(achieved_goal, desired_goal, obs, action, info):
    distance = np.linalg.norm(achieved_goal - desired_goal)
    success_threshold = 0.05
    reward_scale = 10.0  # Scale to make shaping rewards more impactful
    success_bonus = 50.0  # Stronger incentive for success
    if distance < success_threshold:
        return success_bonus  # Higher reward for achieving the goal
    return -reward_scale * distance  # Scaled distance-based shaping reward