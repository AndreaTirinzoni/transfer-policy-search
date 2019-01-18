import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct
from envs.rkhs_env import RKHS_Env

class ModelEstimatorRKHS:
    """
    Model estimation algorithm using reproducing kernel Hilbert spaces.
    """

    def __init__(self, kernel_rho, kernel_lambda, sigma_env, sigma_pi, T, R, lambda_, source_envs, n_source, max_gp, state_dim, action_dim=1,
                 use_gp=False, linear_kernel=False, use_gp_generate_mixture=False, alpha_gp=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.T = T
        self.R = R  # Number of trajectories to approximate the objective
        self.lambda_ = lambda_
        self.source_envs = source_envs
        self.sigma_env = sigma_env
        self.sigma_pi = sigma_pi
        self.n_source = np.array(n_source)
        self.max_gp = max_gp

        if linear_kernel:
            self.kernel = DotProduct(sigma_0=0)
        else:
            self.kernel = kernel_rho**2 * RBF(length_scale=kernel_lambda)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=sigma_env**2 if alpha_gp is None else alpha_gp, optimizer=None)

        # Weight matrix of the learned model
        self.A = 0
        # State-action matrix used to learn the current model
        self.X = 0

        # Env used to simulate trajectories from the learned model
        self.rkhs_env = RKHS_Env(self, source_envs[0], sigma_env)

        # Whether the current GP model should be used to simulate trajectories
        self.use_gp = use_gp

        # Whether the model has been fitted once
        self.model_fitted = False

        # Whether the GP has been fitted once
        self.gp_fitted = False

        # Whether the model has been fitted once
        self.use_gp_generate_mixture = use_gp_generate_mixture

    def _split_dataset(self, dataset):
        """
        Extracts the relevant data structures from a dataset of trajectories.
        """
        # States from the target task
        s_t = dataset.source_task[dataset.initial_size:, :, 0:self.state_dim]
        # Clipped actions from the target task
        a_t = dataset.clipped_actions[dataset.initial_size:, :, np.newaxis]
        sa_t = np.concatenate([s_t, a_t], axis=2)
        # State-action matrix (NTxd+1)
        X = sa_t.reshape(sa_t.shape[0]*sa_t.shape[1], sa_t.shape[2])

        # Next states from the target task
        s_t1 = dataset.next_states_unclipped[dataset.initial_size:, :, :]
        # Next-state matrix (NTxd)
        Y = s_t1.reshape(s_t1.shape[0]*s_t1.shape[1], s_t1.shape[2])

        # f(s,a) from the target task
        f_t = dataset.next_states_unclipped_denoised[dataset.initial_size:, :, :]
        # Transition function at each state-action pair (NTxd)
        F = f_t.reshape(f_t.shape[0]*f_t.shape[1], f_t.shape[2])

        # Handle variable-length trajectories
        trajectories_length = dataset.source_param[:, -1]
        mask = trajectories_length[dataset.initial_size:, np.newaxis] >= np.repeat(np.arange(0, s_t.shape[1])[np.newaxis, :],repeats=s_t.shape[0], axis=0)
        # Reshape mask to NT
        mask = mask.reshape(s_t.shape[0]*s_t.shape[1],)
        # Mask matrices
        X = X[mask, :]
        Y = Y[mask, :]
        F = F[mask, :]

        # Limit the number of samples usable by GPs
        if X.shape[0] > self.max_gp:
            X = X[-self.max_gp:, :]
            Y = Y[-self.max_gp:, :]
            F = F[-self.max_gp:, :]

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

    def eval_model(self, dataset):
        """
        Evaluates the current RKHS model on a dataset of target trajectories. Returns the estimated MSE.
        """
        X, _, F = self._split_dataset(dataset)
        return np.mean((np.matmul(self.kernel(self.X, X).T, self.A) - F)**2)

    def transition(self, state, action):
        """
        Simulates one or more transitions using the current model.

        :return: the estimated transition f(s,a) if a single state and action are passed.
        If state is an NxTxd matrix and action is an NxT matrix, it returns the NxTxd matrix containing all transitions.
        """

        if not self.gp_fitted:
            return np.zeros(state.shape)
        elif state.ndim == 1:
            x = np.append(state, action)[np.newaxis, :]
            if self.use_gp or not self.model_fitted:
                return self.gp.predict(x).reshape(self.state_dim, )
            else:
                return np.matmul(self.kernel(self.X, x).T, self.A).reshape(self.state_dim, )
        else:
            X = np.concatenate([state, action[:,:,np.newaxis]], axis=2).reshape(state.shape[0] * state.shape[1], self.state_dim+1)
            if self.use_gp or not self.model_fitted:
                return self.gp.predict(X).reshape(state.shape)
            else:
                return np.matmul(self.kernel(self.X, X).T, self.A).reshape(state.shape)

    def _update_weight_matrix(self, X, M, W, F_src, alpha_src, c1, c2):
        """
        Updates the current weight matrix of the learned transition model.
        """

        K = self.kernel(X)  # Gram matrix
        # For loop here is faster than concatenating and summing
        F_bar = alpha_src[0] * F_src[0]
        for j in range(len(F_src) - 1):
            F_bar += alpha_src[j + 1] * F_src[j + 1]
        inv = np.linalg.pinv(c1*np.sum(alpha_src)*np.matmul(W, K) + c2*K + self.lambda_*self.R*np.eye(X.shape[0]))
        self.A = np.matmul(inv, c1*np.matmul(W, F_bar) + c2*M)
        self.X = X

    def _collect_dataset_update(self, policy_params, alpha, target_param):
        """
        Collects a dataset for updating the transition model. The collection is under a mixture distribution of
        given policies and the previous learned model. This function also the weight function W to be used in
        fitting the learned model.

        If use_gp is true, the current GP model is used to simulate transitions instead of the old learned model.
        """

        policy_params = np.array(policy_params)
        alpha = np.array(alpha)

        states = np.zeros((self.R, self.T, self.state_dim))
        actions = np.zeros((self.R, self.T))
        probs_mixture = np.ones((self.R, self.T, policy_params.shape[0]))
        probs_target = np.ones((self.R, self.T))

        for r in range(self.R):

            # Choose a policy to run with probabilities alpha
            theta = policy_params[np.random.choice(len(policy_params), p=alpha)]

            s = self.rkhs_env.reset()

            for t in range(self.T):

                a = np.random.normal(np.dot(theta, s), self.sigma_pi)
                ns, _, _, _, a_clipped, _ = self.rkhs_env.step(a)

                states[r, t, :] = s
                # We save the clipped action for learning the model
                actions[r, t] = a_clipped

                probs_target[r, t] = np.exp(-(np.dot(target_param, s) - a)**2 / (2*self.sigma_pi**2))
                probs_mixture[r, t, :] = np.exp(-(np.dot(policy_params, s) - a)**2 / (2*self.sigma_pi**2))

                s = ns

        probs_target = np.cumprod(probs_target, axis=1).reshape(self.R*self.T,)
        probs_mixture = np.sum(np.cumprod(probs_mixture, axis=1) * alpha[np.newaxis, np.newaxis, :], axis=2).reshape(self.R*self.T,)
        W = np.diag(probs_target / probs_mixture)
        X = np.concatenate([states, actions[:,:,np.newaxis]], axis=2).reshape(self.R*self.T, self.state_dim+1)

        return X, W, states, actions

    def update_model(self, dataset):
        """
        Updates the current transition model.

        :param N: total number of trajectories in the current dataset
        :param alpha_tgt: proportion of trajectories from each target policy
        :param alpha_src: proportion of trajectories from each source env
        :param alpha_0: proportion of trajectories from the current target policy
        :param policy_params: list of parameters of all target policies
        :param target_param: current target policy parameters
        """

        self.update_gp(dataset)
        self.gp_fitted = True

        N = dataset.source_param.shape[0]
        alpha_0 = dataset.episodes_per_config[-1] / N
        alpha_tgt = dataset.episodes_per_config[dataset.n_config_cv:] / N
        alpha_src = self.n_source / N

        target_param = dataset.source_param[-1, 1:1+self.state_dim]

        param_indices = np.concatenate(([0], np.cumsum(dataset.episodes_per_config[:-1])))[dataset.n_config_cv:]
        policy_params = dataset.source_param[param_indices, 1:1+self.state_dim]

        if self.use_gp_generate_mixture:
            self.use_gp = True

        X, W, states, actions = self._collect_dataset_update(policy_params, alpha_tgt / np.sum(alpha_tgt), target_param)

        if self.use_gp_generate_mixture:
            self.use_gp = False

        C = 5
        if C <= (1 - alpha_0) / (2 * alpha_0):
            u_alpha = 2 * C * (1 - alpha_0) ** 2 / (alpha_0 * C + 1 - alpha_0) ** 3
        else:
            u_alpha = 8 / (27 * alpha_0)

        c1 = u_alpha / (2 * self.sigma_env**2 * N * (1 - alpha_0))
        c2 = 4 * np.sum(alpha_tgt) / (self.sigma_env**2)  # TODO alpha_0 has been neglected

        M = self.gp.predict(X)
        F_src = [env.stepDenoisedCurrent(states, actions).reshape(X.shape[0], self.state_dim) for env in self.source_envs]

        self._update_weight_matrix(X, M, W, F_src, alpha_src, c1, c2)

        self.model_fitted = True



