def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Distance to target
    distance_to_target = np.linalg.norm(achieved_goal - desired_goal)
    
    # Success threshold
    success_threshold = 0.05
    
    # Reward components
    success_reward = 0.0
    if distance_to_target < success_threshold:
        success_reward = 10.0  # Large bonus for successful placement
    
    # Shaping reward: negative distance to encourage moving closer
    shaping_reward = -distance_to_target
    
    # Intermediate milestone: lifting the block off the table
    block_height = obs[3]  # Assuming 3rd index in 'observation' contains block's height
    table_height = 0.05    # Example table height
    lift_bonus = 2.0 if block_height > table_height + 0.02 else 0.0
    
    # Combine rewards
    reward = shaping_reward + success_reward + lift_bonus
    return reward