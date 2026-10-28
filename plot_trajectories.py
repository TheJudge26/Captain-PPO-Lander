import os
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

def get_trajectory(model_path: str, seed: int = 42):
    print(f"Recording trajectory for {model_path}...")
    model = PPO.load(model_path)
    env = gym.make("LunarLander-v3")
    obs, info = env.reset(seed=seed)

    xs = [obs[0]]
    ys = [obs[1]]
    vys = [obs[3]]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        xs.append(obs[0])
        ys.append(obs[1])
        vys.append(obs[3])
        done = terminated or truncated

    env.close()
    return xs, ys, vys

def main():
    os.makedirs("experiments", exist_ok=True)

    models = {
        "Baseline (PPO)": ("models/baseline/final_model.zip", "blue", "-"),
        "Hacked (Hover Exploit)": ("models/hacked/final_model.zip", "red", "--"),
        "PBRS (Ng et al. 1999)": ("models/pbrs/final_model.zip", "green", "-."),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, (path, color, style) in models.items():
        xs, ys, vys = get_trajectory(path, seed=42)

        ax1.plot(xs, ys, label=name, color=color, linestyle=style, linewidth=2)

        ax2.plot(vys, ys, label=name, color=color, linestyle=style, linewidth=2)

    ax1.axhline(0, color="gray", linestyle=":", label="Surface")
    ax1.scatter([0], [0], color="gold", s=150, marker="*", label="Pad Center (0, 0)")
    ax1.set_title("2D Flight Path (X vs Y)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Horizontal Position (X)")
    ax1.set_ylabel("Altitude (Y)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.axvline(0, color="black", linestyle=":", alpha=0.6)
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_title("Phase Portrait: Altitude vs. Vertical Velocity", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Vertical Velocity (Vy) [Negative = Falling]")
    ax2.set_ylabel("Altitude (Y)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    output_path = "experiments/trajectory_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved trajectory comparison plot to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()