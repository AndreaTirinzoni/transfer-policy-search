import gym
import numpy as np

"""
A gym environment embedding learned transition functions using GPs/RKHS.

This class can be used to simulate trajectories from learned models.
"""

class RKHS_Env(gym.Env):

    def __init__(self, rkhs_model, base_env, sigma_noise):
        self.rkhs_model = rkhs_model
        self.base_env = base_env
        self.sigma_noise = sigma_noise

        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space

    def step(self, action):

        action_clipped = self.base_env.clip_action(action)
        noise = np.random.randn() * self.sigma_noise

        next_state_deterministic = self.rkhs_model.transition(self.state, action_clipped)
        next_state_unclipped = next_state_deterministic + noise
        next_state_clipped = self.base_env.clip_state(next_state_unclipped)

        self.state = next_state_clipped

        reward = 0  # TODO we can also call the base_env to get the actual reward, but it's useless here

        return self.get_state(), reward, False, next_state_unclipped, action_clipped, next_state_deterministic

    def reset(self, state=None):
        self.state = self.base_env.reset()
        return self.get_state()

    def get_state(self):
        return np.array(self.state)
