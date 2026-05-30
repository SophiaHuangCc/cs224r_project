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
    lift_bonus = 1.0 if block_height > table_height + 0.03 else 0.0  # Bonus for lifting > 3cm above the table
    
    # Success bonus
    success_bonus = 10.0 if success else 0.0
    
    # Combine all reward components
    reward += lift_bonus + success_bonus
    
    return reward