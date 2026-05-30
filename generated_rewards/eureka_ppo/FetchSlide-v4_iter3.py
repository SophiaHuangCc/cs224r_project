def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Compute the distance between the puck's current position and the target position
    distance_to_target = np.linalg.norm(achieved_goal - desired_goal)
    
    # Linear distance shaping reward: Stronger gradient as the puck gets closer to the target
    shaping_reward = -distance_to_target
    
    # Success bonus: Provide a large reward when the puck reaches the target position
    success_threshold = 0.05  # Tolerance for considering the puck at the target
    success_bonus = 0.0
    if distance_to_target < success_threshold:
        success_bonus = 100.0  # Increased reward for completing the task
    
    # Penalize episode length to encourage faster success
    time_penalty = -0.1  # Small penalty per timestep
    
    # Combine rewards
    total_reward = shaping_reward + success_bonus + time_penalty
    
    return total_reward