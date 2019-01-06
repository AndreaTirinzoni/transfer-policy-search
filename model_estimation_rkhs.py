import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class ModelEstimatorRKHS:
    """
    Model estimation algorithm using reproducing kernel Hilbert spaces.
    """

    def __init__(self, kernel_rho, kernel_lambda, sigma_env, T, R, lambda_, source_envs, state_dim, action_dim=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.T = T
        self.R = R  # Number of trajectories to approximate the objective
        self.lambda_ = lambda_
        self.source_envs = source_envs
        self.sigma_env = sigma_env

        self.kernel = kernel_rho**2 * RBF(length_scale=kernel_lambda)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=sigma_env**2, optimizer=None)

        # Weight matrix of the learned model
        self.A = 0

    def _split_dataset(self, dataset):
        """
        Extracts the relevant data structures from a dataset of trajectories.
        """
        s_t = dataset.source_task[:, :, 0:self.state_dim]
        # TODO should we use unclipped actions?
        a_t = dataset.clipped_actions[:, :, np.newaxis]
        sa_t = np.concatenate([s_t, a_t], axis=2)
        # State-action matrix (NTxd+1)
        X = sa_t.reshape(sa_t.shape[0]*sa_t.shape[1], sa_t.shape[2])

        # TODO should we use clipped next states?
        s_t1 = dataset.next_states_unclipped
        # Next-state matrix (NTxd)
        Y = s_t1.reshape(s_t1.shape[0]*s_t1.shape[1], s_t1.shape[2])

        f_t = dataset.next_states_unclipped_denoised
        # Transition function at each state-action pair (NTxd)
        F = f_t.reshape(f_t.shape[0]*f_t.shape[1], f_t.shape[2])

        return X, Y, F

    def update_gp(self, dataset):
        """
        Updates the current GP model using a dataset of target trajectories.
        """
        X, Y, _ = self._split_dataset(dataset)
        self.gp.fit(X, Y)

    def eval_gp(self, dataset):
        """
        Evaluates the current GP model on a dataset of target trajectories. Returns the estimated MSE.
        """
        X, _, F = self._split_dataset(dataset)
        return np.mean((self.gp.predict(X) - F)**2)

    def _update_weight_matrix(self, X, M, W, F_src, alpha_src, c1, c2):
        """
        Updates the current weight matrix of the learned transition model.
        """

        K = self.kernel(X)  # Gram matrix
        # For loop here is faster than concatenating and summing
        F_bar = alpha_src[0] * F_src[0]
        for j in range(M - 1):
            F_bar += alpha_src[j + 1] * F_src[j + 1]
        inv = np.linalg.inv(np.matmul(c1*np.sum(alpha_src)*W + c2, K) + self.lambda_*self.R*np.eye(X.shape[0]))
        self.A = np.matmul(inv, c1*np.matmul(W, F_bar) + c2*M)

    def _collect_dataset_update(self):
        """
        Collects a dataset for updating the transition model. The collection is under a mixture distribution of
        given policies and the previous learned model.
        """

        raise NotImplementedError

    def _compute_iw_update(self):
        """
        Computes the importance-weight matrix for updating the transition model.
        """

        raise NotImplementedError

    def update_model(self, ):
        """
        Updates the current transition model.
        """

        dataset = self._collect_dataset_update()

        X, _, _ = self._split_dataset(dataset)

        alpha_0 = 0  # TODO
        alpha_tgt = 0  # TODO
        alpha_src = [0]  # TODO
        N = 0  # TODO

        C_alpha = (1 - alpha_0)^2 / (alpha_0 * np.log(1 / alpha_0))
        c1 = C_alpha / (2 * self.sigma_env**2 * N)
        c2 = 4 * alpha_tgt / (self.sigma_env**2 * alpha_0**2)

        M = self.gp.predict(X)
        W = self._compute_iw_update()
        F_src = [env.stepDenoisedCurrent(dataset.source_task[:, :, 0:self.state_dim],
                                         dataset.clipped_actions).reshape(X.shape[0], self.state_dim) for env in self.source_envs]

        self._update_weight_matrix(X, M, W, F_src, alpha_src, c1, c2)



