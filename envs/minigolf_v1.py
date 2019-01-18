import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math as m

"""
Minigolf task.

References
----------
  - Penner, A. R. "The physics of putting." Canadian Journal of Physics 80.2 (2002): 83-96.

"""

class MiniGolf(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.horizon = 20
        self.gamma = 0.99

        self.min_pos = 0.0
        self.max_pos = 20.0
        self.min_action = 0.0
        self.max_action = 10.0
        self.putter_length = 1.0 # [0.7:1.0]
        self.friction = 0.131 # [0.065:0.196]
        self.hole_size = 0.10 # [0.10:0.15]
        self.sigma_noise = 0.3
        self.ball_radius = 0.02135

        # gym attributes
        self.viewer = None
        low = np.array([self.min_pos])
        high = np.array([self.max_pos])
        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action,
                                       shape=(1,))
        self.observation_space = spaces.Box(low=low, high=high)

        # initialize state
        self.seed()
        self.reset()

    def setParams(self, env_param):
        self.putter_length = env_param[0]
        self.friction = env_param[1]
        self.hole_size = env_param[2]
        self.sigma_noise = m.sqrt(env_param[-1])

    def step(self, action, render=False):
        noise = self.np_random.randn() * self.sigma_noise
        action_clipped = np.clip(action, self.min_action, self.max_action)
        u = action_clipped * self.putter_length

        # v_min = np.sqrt(10 / 7 * self.friction * 9.81 * self.state)
        # v_max = np.sqrt((2*self.hole_size - self.ball_radius)**2*(9.81/(2*self.ball_radius)) + v_min**2)

        deceleration = 5 / 7 * self.friction * 9.81
        t = u / deceleration
        xn_unclipped_denoised = self.state - u * t + 0.5 * deceleration * t ** 2
        xn_unclipped = xn_unclipped_denoised + noise

        if xn_unclipped < 0:
            reward = -100
            done = True
        elif xn_unclipped < self.hole_size:
            reward = 0
            done = True
        else:
            done = False
            reward = -1

        xn = np.clip(xn_unclipped, self.min_pos, self.max_pos)

        self.state = xn

        # We return the unclipped state and the clipped action as the last argument (to be used for computing the importance weights only)
        return self.get_state(), float(reward), done, xn_unclipped, action_clipped, xn_unclipped_denoised

    #Custom param for transfer

    def getEnvParam(self):
        return np.asarray([self.putter_length, self.friction, self.hole_size, self.sigma_noise])

    def reset(self, state=None):
        if state is None:
            self.state = np.array([self.np_random.uniform(low=self.min_pos,
                                                          high=self.max_pos)])
        else:
            self.state = np.array(state)

        return self.get_state()

    def get_state(self):
        return np.array(self.state)

    def clip_state(self, state):
        return np.clip(state, self.min_pos, self.max_pos)

    def clip_action(self, action):
        return np.clip(action, self.min_action, self.max_action)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]