def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Distance to target
    distance_to_target = np.linalg.norm(achieved_goal - desired_goal)
    
    # Success threshold
    success_threshold = 0.05
    
    # Reward components
    success_reward = 0.0
    if distance_to_target < success_threshold:
        success_reward = 50.0  # Larger bonus for successful placement
    
    # Shaping reward: negative distance with stronger emphasis
    shaping_reward = -10.0 * distance_to_target
    
    # Intermediate milestone: lifting the block off the table
    block_height = obs[3]  # Assuming 3rd index in 'observation' contains block's height
    table_height = 0.05    # Example table height
    lift_bonus = 5.0 if block_height > table_height + 0.02 else 0.0
    
    # Penalty for excessive actions to encourage efficiency
    action_penalty = -0.1 * np.linalg.norm(action)
    
    # Combine rewards
    reward = shaping_reward + success_reward + lift_bonus + action_penalty
    return reward