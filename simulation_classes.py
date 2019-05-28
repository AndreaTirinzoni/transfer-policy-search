class EnvParam:
    """
    Information about the environment
    """
    def __init__(self, env, param_space_size, state_space_size, env_param_space_size, episode_length, gaussian_transition):
        self.env = env
        self.param_space_size = param_space_size
        self.state_space_size = state_space_size
        self.env_param_space_size = env_param_space_size
        self.episode_length = episode_length
        self.gaussian_transition = gaussian_transition


class SimulationParam:
    """
    Informations about the parameters used in the simulation
    """
    def __init__(self, mean_initial_param, variance_initial_param, variance_action, batch_size, num_batch, discount_factor, runs, learning_rate, ess_min, adaptive, defensive_sample, use_adam=False):
        self.mean_initial_param = mean_initial_param
        self.variance_initial_param = variance_initial_param
        self.variance_action = variance_action
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.discount_factor = discount_factor
        self.runs = runs
        self.learning_rate = learning_rate
        self.ess_min = ess_min
        self.adaptive = adaptive
        self.defensive_sample = defensive_sample
        self.use_adam = use_adam


class SourceDataset:
    """
    Informations related to the episodes in the source dataset
    """
    def __init__(self, source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, next_states_unclipped_denoised, n_config_src):
        self.source_task = source_task
        self.source_param = source_param
        self.episodes_per_config = episodes_per_config
        self.next_states_unclipped = next_states_unclipped
        self.next_states_unclipped_denoised = next_states_unclipped_denoised
        self.clipped_actions = clipped_actions
        self.n_config_src = n_config_src
        self.initial_size = source_task.shape[0]
        self.source_distributions = None
        self.mask_weights = None
        self.source_distributions_cv = None
        self.policy_per_model = None
