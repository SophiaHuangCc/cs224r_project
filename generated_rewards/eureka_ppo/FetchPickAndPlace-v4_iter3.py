def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Distance to the target position
    distance = np.linalg.norm(achieved_goal - desired_goal)
    
    # Success condition
    success_threshold = 0.05  # Success if within 5cm of the target
    success = distance < success_threshold
    
    # Reward shaping: negative distance as the primary term
    reward = -distance
    
    # Stronger incentive for reducing distance to the target
    distance_bonus = 1 / (distance + 1e-6)  # Higher reward as distance decreases
    
    # Reward for lifting the block off the table
    block_height = achieved_goal[2]
    table_height = 0.02  # Assume table height is 2cm
    lift_bonus = 1.0 * max(0, block_height - (table_height + 0.03))  # Reward for lifting > 3cm above the table
    
    # Penalize episode length to encourage faster convergence
    step_penalty = -0.05  # Stronger penalty for longer episodes
    
    # Success bonus
    success_bonus = 50.0 if success else 0.0  # Higher success bonus
    
    # Combine all reward components
    reward += distance_bonus + lift_bonus + success_bonus + step_penalty
    
    return reward