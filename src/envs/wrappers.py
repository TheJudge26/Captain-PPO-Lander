import gymnasium as gym

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