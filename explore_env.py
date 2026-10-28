import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset(seed=42)
print("Initial 8 nums:", observation)

done = False
while not done:
    
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Action: {action} | Height: {observation[1]:.2f} | Reward: {reward:.2f}")

    done = terminated or truncated

env.close()