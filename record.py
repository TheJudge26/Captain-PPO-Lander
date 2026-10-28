import os
import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Record trained agent flight")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml")
    parser.add_argument("--model-path", type=str, default="models/baseline/final_model.zip")
    parser.add_argument("--video-dir", type=str, default="videos/baseline")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)

    env = gym.make(
        config["env"]["id"],
        render_mode="rgb_array",
        continuous=config["env"]["continuous"],
        gravity=config["env"]["gravity"],
        enable_wind=config["env"]["enable_wind"]
    )

    env = RecordVideo(
        env,
        video_folder=args.video_dir,
        episode_trigger=lambda ep_id: True, 
        name_prefix="baseline_landing"
    )

    print(f"Loading trained model from {args.model_path}...")
    model = PPO.load(args.model_path)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep + 100)
        done = False
        total_reward = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Flight {ep + 1}: Total Score = {total_reward:.2f}")

    env.close()
    print(f"All recordings saved to folder: {args.video_dir}")

if __name__ == "__main__":
    main()