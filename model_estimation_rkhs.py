import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class ModelEstimatorRKHS:
    """
    Model estimation algorithm using reproducing kernel Hilbert spaces.
    """

    def __init__(self, kernel_rho, kernel_lambda, sigma_env, T, state_dim, action_dim=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.T = T

        self.kernel = kernel_rho**2 * RBF(length_scale=kernel_lambda)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=sigma_env**2, optimizer=None)

    def update_gp(self, dataset):
        """
        Updates the current GP model using a dataset of target trajectories.
        """
        # States for each trajectory and time (NxTxd)
        s_t = dataset.source_task[:, :, 0:self.state_dim]
        # TODO should we use unclipped actions?
        a_t = dataset.clipped_actions[:, :, np.newaxis]
        sa_t = np.concatenate([s_t, a_t], axis=2)
        # Input matrix (NTxd+1)
        X = sa_t.reshape(sa_t.shape[0]*sa_t.shape[1], sa_t.shape[2])

        # TODO should we use clipped next states?
        s_t1 = dataset.next_states_unclipped
        # Output matrix (NTxd)
        Y = s_t1.reshape(s_t1.shape[0]*s_t1.shape[1], s_t1.shape[2])

        self.gp.fit(X, Y)

    def eval_gp(self, dataset):
        """
        Evaluates the current GP model on a dataset of target trajectories. Returns the estimated MSE.
        """
        # States for each trajectory and time (NxTxd)
        s_t = dataset.source_task[:, :, 0:self.state_dim]
        # TODO should we use unclipped actions?
        a_t = dataset.clipped_actions[:, :, np.newaxis]
        sa_t = np.concatenate([s_t, a_t], axis=2)
        # Input matrix (NTxd+1)
        X = sa_t.reshape(sa_t.shape[0]*sa_t.shape[1], sa_t.shape[2])

        s_t1 = dataset.next_states_unclipped_denoised
        Y = s_t1.reshape(s_t1.shape[0]*s_t1.shape[1], s_t1.shape[2])

        return np.mean((self.gp.predict(X) - Y)**2)