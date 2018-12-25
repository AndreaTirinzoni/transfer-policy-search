import gym
from gym import spaces
import numpy as np

"""
A simple test environment.

State space: 2D in [-1,1]
Action space: 1D in [-1,1]
Env param (omega): 1D in [0,1]

Transitions: f(s,a) = -sign(s)*omega, sigma = 1
Reward: R(s,a) = -|a|
Initial state: random in {[-1,-1],[1,-1],[-1,1],[1,1]}*omega
Horizon: 2
gamma: 1

Optimal policy: theta = [0,0]
"""

class TestEnv(gym.Env):

    def __init__(self):
        self.horizon = 2
        self.gamma = 1
        self.state_dim = 2

        self.sigma_noise = 1
        self.omega = 1
        self.deterministic = True  # Whether to add noise to the transitions or not

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,))

    def setParams(self, env_param):
        assert env_param.shape == (2,)
        self.omega = env_param[0]
        self.sigma_noise = np.sqrt(env_param[1])

    def step(self, action):
        action_clipped = np.clip(action, -1, 1)
        noise = np.random.randn() * self.sigma_noise

        next_state_deterministic = -np.sign(self.state) * self.omega
        next_state_unclipped = next_state_deterministic if self.deterministic else next_state_deterministic + noise
        next_state_clipped = np.clip(next_state_unclipped, -1, 1)

        self.state = next_state_clipped

        reward = -np.asscalar(action_clipped ** 2)

        return self.get_state(), reward, False, next_state_unclipped, action_clipped, next_state_deterministic

    def getEnvParam(self):
        return np.array([self.omega, self.sigma_noise**2])

    def stepDenoised(self, env_parameters, state, action):
        omegas = env_parameters[:, 0].squeeze()
        return -np.sign(state) * omegas[np.newaxis, np.newaxis, np.newaxis, :]

    def stepDenoisedCurrent(self, state, action):
        return -np.sign(state) * self.omega

    def reset(self, state=None):
        if state is None:
            self.state = np.random.choice([-1, 1], size=(2,)) * self.omega
        else:
            self.state = np.array(state)

        return self.get_state()

    def get_state(self):
        return np.array(self.state)