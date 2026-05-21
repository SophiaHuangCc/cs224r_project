def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Compute the distance between the puck's current position and the target position
    distance_to_target = np.linalg.norm(achieved_goal - desired_goal)
    
    # Exponential distance shaping reward: Stronger gradient as the puck gets closer to the target
    shaping_reward = -np.exp(distance_to_target)
    
    # Success bonus: Provide a large reward when the puck reaches the target position
    success_threshold = 0.05  # Tolerance for considering the puck at the target
    success_bonus = 0.0
    if distance_to_target < success_threshold:
        success_bonus = 50.0  # Reward for completing the task
    
    # Combine shaping reward and success bonus
    total_reward = shaping_reward + success_bonus
    
    return total_reward