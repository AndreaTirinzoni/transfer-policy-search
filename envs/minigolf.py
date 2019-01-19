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

        if state < next_state:
            return 0

        action = np.clip(action, self.min_action, self.max_action / 2)

        putter_length = env_parameters[0]
        friction = env_parameters[1]
        sigma_noise = env_parameters[-1]
        deceleration = 5 / 7 * friction * 9.81

        u = np.sqrt(2 * deceleration * (state - next_state))
        noise = (u / (action*putter_length) - 1) / sigma_noise
        noise[np.isnan(noise)] = 1e-8

        return norm.pdf(noise)

    def density(self, env_parameters, state, action, next_state):
        """

        :param env_parameters: list of env_params
        :param state: NxTx1
        :param action: NxT
        :param next_state: NxTx1
        :return: pdf NxTx1xn_param
        """
        assert state.ndim == 4 and action.ndim == 3 and next_state.ndim == 4

        mask = state < next_state
        action = np.clip(action, self.min_action, self.max_action / 2)
        pdf = np.zeros((state.shape[0], state.shape[1], 1, env_parameters.shape[0]))
        diff = np.abs(state - next_state)  # take the abs for the sqrt, but mask negative values later

        for i in range(env_parameters.shape[0]):
            deceleration = 5 / 7 * env_parameters[i, 1] * 9.81
            u = np.sqrt(2 * deceleration * diff[:, :, :, i])
            noise = (u / (action * env_parameters[i, 0]) - 1) / env_parameters[i, -1]
            noise[np.isnan(noise)] = 1e-8
            pdf[:, :, :, i] = norm.pdf(noise) * (1-mask[:, :, :, i])  # set to zero impossible transitions

        return pdf[:, :, 0, :]

    def densityCurrent(self, state, action, next_state):
        """
        :param state: NxTx1
        :param action: NxT
        :param next_state: NxTx1
        :return: pdf NxTx1xn_param
        """

        assert state.ndim == 3 and action.ndim == 2 and next_state.ndim == 3

        mask = state < next_state
        action = np.clip(action, self.min_action, self.max_action / 2)
        diff = np.abs(state - next_state)  # take the abs for the sqrt, but mask negative values later

        deceleration = 5 / 7 * self.putter_length * 9.81
        u = np.sqrt(2 * deceleration * diff)
        noise = (u / (action[:, :, np.newaxis] * self.friction) - 1) / self.sigma_noise
        noise[np.isnan(noise)] = 1e-8
        pdf = norm.pdf(noise) * (1-mask)  # set to zero impossible transitions

        return pdf[:, :, 0]
