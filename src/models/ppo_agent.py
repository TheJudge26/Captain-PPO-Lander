import gymnasium as gym
from stable_baselines3 import PPO

def create_ppo_agent(env: gym.Env, config: dict) -> PPO:
    ppo_cfg = config["ppo"]
    training_cfg = config["training"]

    model = PPO(
    policy=ppo_cfg["policy"],             
    env=env,
    learning_rate=float(ppo_cfg["learning_rate"]), 
    n_steps=int(ppo_cfg["n_steps"]),             
    batch_size=int(ppo_cfg["batch_size"]),        
    n_epochs=int(ppo_cfg["n_epochs"]),            
    gamma=float(ppo_cfg["gamma"]),                
    gae_lambda=float(ppo_cfg["gae_lambda"]),      
    clip_range=float(ppo_cfg["clip_range"]),      
    ent_coef=float(ppo_cfg["ent_coef"]),         
    verbose=1,                                    
    tensorboard_log=training_cfg["log_dir"],      
    seed=training_cfg["seed"]                     
)
    return model