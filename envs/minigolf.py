"""classic Linear Quadratic Gaussian Regulator task"""
from numbers import Number

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math as m
from scipy.stats import norm

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
        action = np.clip(action, self.min_action, self.max_action / 2)

        noise = 10
        while abs(noise) > 1:
            noise = self.np_random.randn() * self.sigma_noise
        u = action * self.putter_length * (1 + noise)

        v_min = np.sqrt(10 / 7 * self.friction * 9.81 * self.state)
        v_max = np.sqrt((2*self.hole_size - self.ball_radius)**2*(9.81/(2*self.ball_radius)) + v_min**2)

        deceleration = 5 / 7 * self.friction * 9.81

        t = u / deceleration
        xn = self.state - u * t + 0.5 * deceleration * t ** 2

        reward = 0
        done = True
        if u < v_min:
            reward = -1
            done = False
        elif u > v_max:
            reward = -100

        self.state = xn

        # TODO the last three values should not be used
        return self.get_state(), float(reward), done, xn, action, xn

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
        #return np.array([np.asscalar(self.state)**i/20**i for i in range(6)])
        return np.array(self.state)

    def get_true_state(self):
        """For testing purposes"""
        return np.array(self.state)

    def clip_state(self, state):
        return np.clip(state, self.min_pos, self.max_pos)

    def clip_action(self, action):
        return np.clip(action, self.min_action, self.max_action)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getDensity(self, env_parameters, state, action, next_state):
        putter_length = env_parameters[0]
        friction = env_parameters[1]
        sigma_noise = env_parameters[-1]
        deceleration = 5 / 7 * friction * 9.81
        u = np.sqrt(2 * deceleration * (state - next_state))
        noise = (u / (action*putter_length) - 1) / sigma_noise
        return norm.pdf(noise)

