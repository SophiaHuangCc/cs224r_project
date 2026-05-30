def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Distance to the target position
    distance = np.linalg.norm(achieved_goal - desired_goal)
    
    # Success condition
    success_threshold = 0.05  # Success if within 5cm of the target
    success = distance < success_threshold
    
    # Reward shaping: negative distance as the primary term
    reward = -distance
    
    # Intermediate milestone: reward for lifting the block off the table
    block_height = achieved_goal[2]
    table_height = 0.02  # Assume table height is 2cm
    lift_bonus = 0.5 * max(0, block_height - (table_height + 0.03))  # Gradual reward for lifting > 3cm above the table
    
    # Penalize episode length to encourage faster convergence
    step_penalty = -0.01
    
    # Success bonus
    success_bonus = 20.0 if success else 0.0
    
    # Combine all reward components
    reward += lift_bonus + success_bonus + step_penalty
    
    return reward