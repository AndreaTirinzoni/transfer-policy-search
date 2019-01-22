import gym
import numpy as np

"""
A gym environment embedding learned transition functions using estimated models.

This class can be used to simulate trajectories from learned models in case of planning problems
"""

class Planning_env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, transition_model, base_env, sigma_noise):
        self.transition_model = transition_model
        self.base_env = base_env
        self.sigma_noise = sigma_noise
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space

    def step(self, action):

        action_clipped = self.base_env.clip_action(action)
        noise = np.random.randn() * self.sigma_noise

        next_state_deterministic = self.transition_model.predict(self.state, action_clipped)
        next_state_unclipped = next_state_deterministic + noise
        next_state_clipped = self.base_env.clip_state(next_state_unclipped)

        [reward, done] = self.base_env.reward(self.state, action_clipped, next_state_unclipped)

        self.state = next_state_clipped
        # TODO check

        return self.get_state(), reward, done, next_state_unclipped, action_clipped, next_state_deterministic

    def reset(self, state=None):
        self.state = self.base_env.reset()
        return self.get_state()

    def get_state(self):
        return np.array(self.state)
