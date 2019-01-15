"""classic Linear Quadratic Gaussian Regulator task"""
from numbers import Number

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
        u = np.clip(action*self.putter_length*(1 + noise), self.min_action, self.max_action)

        v_min = np.sqrt(10 / 7 * self.friction * 9.81 * self.state)
        v_max = np.sqrt( (2*self.hole_size - self.ball_radius)**2*(9.81/(2*self.ball_radius)) + v_min**2)

        reward = 0
        done = True
        xn = 0
        if u < v_min:
            deceleration = 5/7*self.friction*9.81
            t = u / deceleration
            xn = self.state - u*t + 0.5*9.81*t**2
            reward = -1
            done = False
        elif u > v_max:
            reward = -100

        self.state = xn

        # We return the unclipped state and the clipped action as the last argument (to be used for computing the importance weights only)
        return self.get_state(), float(reward), done, xn, action, xn

    #Custom param for transfer

    def getEnvParam(self):
        return np.asarray([self.putter_length, self.deceleration, self.hole_size, self.sigma_noise])

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

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 6000
        screen_height = 4000

        world_width = (self.max_pos * 2) * 2
        scale = screen_width / world_width
        bally = 100
        ballradius = 3

        if self.viewer is None:
            clearance = 0  # y-offset
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            mass = rendering.make_circle(ballradius * 2)
            mass.set_color(.8, .3, .3)
            mass.add_attr(rendering.Transform(translation=(0, clearance)))
            self.masstrans = rendering.Transform()
            mass.add_attr(self.masstrans)
            self.viewer.add_geom(mass)
            self.track = rendering.Line((0, bally), (screen_width, bally))
            self.track.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(self.track)
            zero_line = rendering.Line((screen_width / 2, 0),
                                       (screen_width / 2, screen_height))
            zero_line.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(zero_line)

        x = self.state[0]
        ballx = x * scale + screen_width / 2.0
        self.masstrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


