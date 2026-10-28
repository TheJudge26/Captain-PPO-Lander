import os
import argparse
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback

from src.utils.config import load_config
from src.models.ppo_agent import create_ppo_agent
from src.envs.wrappers import NaiveHoverRewardWrapper

def main():
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    os.makedirs(config["training"]["save_dir"], exist_ok=True)
    os.makedirs(config["training"]["log_dir"], exist_ok=True)

    env = gym.make(
        config["env"]["id"],
        continuous=config["env"]["continuous"],
        gravity=config["env"]["gravity"],
        enable_wind=config["env"]["enable_wind"]
    )

    wrapper_type = config["env"].get("wrapper")
    if wrapper_type == "naive_hover":
        bonus = float(config["env"].get("bonus_weight", 1.0))
        print(f"Applying NaiveHoverRewardWrapper (bonus_weight={bonus})...")
        env = NaiveHoverRewardWrapper(env, bonus_weight=bonus)

    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"]["save_freq"],
        save_path=config["training"]["save_dir"],
        name_prefix="ppo_checkpoint"
    )

    model = create_ppo_agent(env, config)

    print(f"Starting training for {config['training']['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=checkpoint_callback
    )

    final_path = os.path.join(config["training"]["save_dir"], "final_model")
    model.save(final_path)
    print(f"Training complete! Final model saved to: {final_path}.zip")
    env.close()

if __name__ == "__main__":
    main()