import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

def evaluate_model(model_path: str, num_episodes: int = 20) -> dict:
    print(f"Evaluating {model_path} across {num_episodes} episodes...")
    model = PPO.load(model_path)
    env = gym.make("LunarLander-v3") 

    rewards = []
    durations = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)  
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        durations.append(steps)

    env.close()

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": np.mean([r >= 200 for r in rewards]) * 100,
        "crash_rate": np.mean([r <= -100 for r in rewards]) * 100,
        "avg_duration": np.mean(durations),
    }

def main():
    models = {
        "Baseline": "models/baseline/final_model.zip",
        "Hacked (Hover Exploit)": "models/hacked/final_model.zip",
        "PBRS (Ng et al. 1999)": "models/pbrs/final_model.zip",
    }

    results = {}
    for name, path in models.items():
        results[name] = evaluate_model(path, num_episodes=20)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON ")
    print("=" * 80)
    print(f"{'Model':<25} | {'Reward (Mean ± Std)':<20} | {'Success':<8} | {'Crash':<8} | {'Avg Steps':<10}")
    print("-" * 80)

    for name, r in results.items():
        reward_str = f"{r['mean_reward']:.1f} ± {r['std_reward']:.1f}"
        print(f"{name:<25} | {reward_str:<20} | {r['success_rate']:>6.1f}% | {r['crash_rate']:>6.1f}% | {r['avg_duration']:>9.1f}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()