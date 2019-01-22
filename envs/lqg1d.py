"""classic Linear Quadratic Gaussian Regulator task"""
from numbers import Number

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math as m

"""
Linear quadratic gaussian regulator task.

References
----------
  - Simone Parisi, Matteo Pirotta, Nicola Smacchia,
    Luca Bascetta, Marcello Restelli,
    Policy gradient approaches for multi-objective sequential decision making
    2014 International Joint Conference on Neural Networks (IJCNN)
  - Jan  Peters  and  Stefan  Schaal,
    Reinforcement  learning of motor  skills  with  policy  gradients,
    Neural  Networks, vol. 21, no. 4, pp. 682-697, 2008.

"""

class LQG1D(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.horizon = 20
        self.gamma = 0.99

        self.max_pos = 10.0
        self.max_action = 8.0
        self.sigma_noise = 0.3
        self.param_dim = 1
        self.A = np.array([1]).reshape((1, 1))
        self.B = np.array([1]).reshape((1, 1))
        self.Q = np.array([0.9]).reshape((1, 1))
        self.R = np.array([0.9]).reshape((1, 1))

        # gym attributes
        self.viewer = None
        high = np.array([self.max_pos])
        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action,
                                       shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        # initialize state
        self.seed()
        self.reset()

    def setParams(self, env_param):
        dim1a = self.A.shape[0]
        dim2a = self.A.shape[1]
        self.A = env_param[0:(dim1a*dim2a)].reshape((dim1a, dim2a))
        dim1b = self.B.shape[0]
        dim2b = self.B.shape[1]
        self.B = env_param[(dim1a*dim2a):(dim1a*dim2a)+(dim1b*dim2b)].reshape((dim1b, dim2b))
        self.sigma_noise = m.sqrt(env_param[-1])

    def step(self, action, render=False):
        u = np.clip(action, -self.max_action, self.max_action)
        noise = self.np_random.randn() * self.sigma_noise
        xn_unclipped = np.dot(self.A, self.state) + np.dot(self.B, u) + noise
        xn_unclipped_denoised = np.dot(self.A, self.state) + np.dot(self.B, u)
        xn = np.clip(xn_unclipped, -self.max_pos, self.max_pos)
        cost = np.dot(self.state, np.dot(self.Q, self.state)) + np.dot(u, np.dot(self.R, u))

        self.state = np.array(xn.ravel())

        # We return the unclipped state and the clipped action as the last argument (to be used for computing the importance weights only)
        return self.get_state(), -np.asscalar(cost), False, np.array(xn_unclipped.ravel()), u, xn_unclipped_denoised

    def reward(self, state, action, next_state):
        u = np.clip(action, -self.max_action, self.max_action)
        cost = np.dot(self.state, np.dot(self.Q, self.state)) + np.dot(u, np.dot(self.R, u))
        return -cost, False

    #Custom param for transfer

    def getEnvParam(self):
        return np.asarray([np.ravel(self.A, order="C"), np.ravel(self.B, order="C"), np.ravel(self.sigma_noise**2, order="C")])

    def stepDenoised(self, env_parameters, state, action):
        num_episodes = env_parameters.shape[0]
        dim1a = self.A.shape[0]
        dim2a = self.A.shape[1]
        A = env_parameters[:, 0:(dim1a*dim2a)].reshape((num_episodes, dim1a, dim2a))
        dim1b = self.B.shape[0]
        dim2b = self.B.shape[1]
        B = env_parameters[:, (dim1a*dim2a):(dim1a*dim2a)+(dim1b*dim2b)].reshape((num_episodes, dim1b, dim2b))
        xn_unclipped = np.sum(np.multiply((A.T)[np.newaxis, np.newaxis, :, :, :], state[:, :, :, np.newaxis, :]), axis=3) + np.multiply(B.T, action)[:, :, np.newaxis, :]
        return xn_unclipped

    def stepDenoisedCurrent(self, state, action):
        A = self.A
        B = self.B
        xn_unclipped = np.sum(np.multiply((A.T)[np.newaxis, np.newaxis, :, :], state[:, :, :, np.newaxis]), axis=2) + np.multiply(B, action)[:, :, np.newaxis]
        return xn_unclipped

    def reset(self, state=None):
        if state is None:
            self.state = np.array([self.np_random.uniform(low=-self.max_pos,
                                                          high=self.max_pos)])
        else:
            self.state = np.array(state)

        return self.get_state()

    def get_state(self):
        return np.array(self.state)

    def clip_state(self, state):
        return np.clip(state, -self.max_pos, self.max_pos)

    def clip_action(self, action):
        return np.clip(action, -self.max_action, self.max_action)

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

    def _computeP2(self, K):
        """
        This function computes the Riccati equation associated to the LQG
        problem.
        Args:
            K (matrix): the matrix associated to the linear controller K * x

        Returns:
            P (matrix): the Riccati Matrix

        """
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if np.array_equal(self.A, I) and np.array_equal(self.B, I):
            P = (self.Q + np.dot(K.T, np.dot(self.R, K))) / (I - self.gamma *
                                                             (I + 2 * K + K **
                                                              2))
        else:
            tolerance = 0.0001
            converged = False
            P = np.eye(self.Q.shape[0], self.Q.shape[1])
            while not converged:
                Pnew = self.Q + self.gamma * np.dot(self.A.T,
                                                    np.dot(P, self.A)) + \
                       self.gamma * np.dot(K.T, np.dot(self.B.T,
                                                       np.dot(P, self.A))) + \
                       self.gamma * np.dot(self.A.T,
                                           np.dot(P, np.dot(self.B, K))) + \
                       self.gamma * np.dot(K.T,
                                           np.dot(self.B.T,
                                                  np.dot(P, np.dot(self.B,
                                                                   K)))) + \
                       np.dot(K.T, np.dot(self.R, K))
                converged = np.max(np.abs(P - Pnew)) < tolerance
                P = Pnew
        return P

    def computeOptimalK(self):
        """
        This function computes the optimal linear controller associated to the
        LQG problem (u = K * x).

        Returns:
            K (matrix): the optimal controller

        """
        P = np.eye(self.Q.shape[0], self.Q.shape[1])
        for i in range(100):
            K = -self.gamma * np.dot(np.linalg.inv(
                self.R + self.gamma * (np.dot(self.B.T, np.dot(P, self.B)))),
                                       np.dot(self.B.T, np.dot(P, self.A)))
            P = self._computeP2(K)
        K = -self.gamma * np.dot(np.linalg.inv(self.R + self.gamma *
                                               (np.dot(self.B.T,
                                                       np.dot(P, self.B)))),
                                 np.dot(self.B.T, np.dot(P, self.A)))
        return K

    def computeJ(self, K, Sigma, n_random_x0=100):
        """
        This function computes the discounted reward associated to the provided
        linear controller (u = Kx + \epsilon, \epsilon \sim N(0,\Sigma)).
        Args:
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
                            the controller action
            n_random_x0: the number of samples to draw in order to average over
                         the initial state

        Returns:
            J (float): The discounted reward

        """
        if isinstance(K, Number):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, Number):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self._computeP2(K)
        J = 0.0
        for i in range(n_random_x0):
            self.reset()
            x0 = self.get_state()
            J -= np.dot(x0.T, np.dot(P, x0)) \
                + (1 / (1 - self.gamma)) * \
                np.trace(np.dot(
                    Sigma, (self.R + self.gamma * np.dot(self.B.T,
                                                         np.dot(P, self.B)))))
        J /= n_random_x0
        return J

    def computeQFunction(self, x, u, K, Sigma, n_random_xn=100):
        """
        This function computes the Q-value of a pair (x,u) given the linear
        controller Kx + epsilon where epsilon \sim N(0, Sigma).
        Args:
            x (int, array): the state
            u (int, array): the action
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
            the controller action
            n_random_xn: the number of samples to draw in order to average over
            the next state

        Returns:
            Qfun (float): The Q-value in the given pair (x,u) under the given
            controller

        """
        if isinstance(x, Number):
            x = np.array([x])
        if isinstance(u, Number):
            u = np.array([u])
        if isinstance(K, Number):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, Number):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self._computeP2(K)
        Qfun = 0
        for i in range(n_random_xn):
            noise = self.np_random.randn() * self.sigma_noise
            action_noise = self.np_random.multivariate_normal(
                np.zeros(Sigma.shape[0]), Sigma, 1)
            nextstate = np.dot(self.A, x) + np.dot(self.B,
                                                   u + action_noise) + noise
            Qfun -= np.dot(x.T, np.dot(self.Q, x)) + \
                np.dot(u.T, np.dot(self.R, u)) + \
                self.gamma * np.dot(nextstate.T, np.dot(P, nextstate)) + \
                (self.gamma / (1 - self.gamma)) * \
                np.trace(np.dot(Sigma,
                                self.R + self.gamma *
                                np.dot(self.B.T, np.dot(P, self.B))))
        Qfun = np.asscalar(Qfun) / n_random_xn
        return Qfun
