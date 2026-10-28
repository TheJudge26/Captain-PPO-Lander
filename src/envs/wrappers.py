import gymnasium as gym
import math

class NaiveHoverRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, bonus_weight: float = 1.0):
        super().__init__(env)
        self.bonus_weight = bonus_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vertical_speed = obs[3] 
        tilt = obs[4]           

        if abs(tilt) < 0.2 and abs(vertical_speed) < 0.2:
            reward += self.bonus_weight

        return obs, reward, terminated, truncated, info

class PBRSRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, gamma: float = 0.99, shaping_weight: float = 1.0):
        super().__init__(env)
        self.gamma = gamma
        self.shaping_weight = shaping_weight
        self.prev_potential = 0.0

    def _compute_potential(self, obs) -> float:
        distance = math.sqrt(obs[0]**2 + obs[1]**2)
        speed = math.sqrt(obs[2]**2 + obs[3]**2)
        tilt = abs(obs[4])

        potential = -(distance + speed + 0.5 * tilt)
        return potential

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_potential = self._compute_potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_potential = self._compute_potential(obs)

        shaping_reward = (self.gamma * current_potential) - self.prev_potential
        reward += self.shaping_weight * shaping_reward

        self.prev_potential = current_potential

        return obs, reward, terminated, truncated, info