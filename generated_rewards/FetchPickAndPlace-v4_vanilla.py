def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Compute Euclidean distance between achieved and desired goals
    distance = np.linalg.norm(achieved_goal - desired_goal)

    # Define a sparse reward: success if within a small threshold
    success_threshold = 0.05
    if distance < success_threshold:
        return 1.0  # Reward for completing the task

    # Define a dense reward: penalize based on distance to encourage efficiency
    dense_reward = -distance

    # Combine sparse and dense rewards
    return dense_reward