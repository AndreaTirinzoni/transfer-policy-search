import gym
import numpy as np
import learningAlgorithm as la
import sourceTaskCreation as stc
from utils import plot

class EnvParam:

    def __init__(self, env, param_space_size, state_space_size, env_param_space_size, episode_length):

        self.env = env
        self.param_space_size = param_space_size
        self.state_space_size = state_space_size
        self.env_param_space_size = env_param_space_size
        self.episode_length = episode_length

class SimulationParam:

    def __init__(self, mean_initial_param, variance_initial_param, variance_action, batch_size, num_batch, discount_factor, runs, learning_rate, ess_min, adaptive):

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

class SourceDataset:

    def __init__(self, source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, next_states_unclipped_denoised):

        self.source_task = source_task
        self.source_param = source_param
        self.episodes_per_config = episodes_per_config
        self.next_states_unclipped = next_states_unclipped
        self.next_states_unclipped_denoised = next_states_unclipped_denoised
        self.clipped_actions = clipped_actions
        self.n_config_cv = episodes_per_config.shape[0]
        self.initial_size = source_task.shape[0]
        self.source_distributions = None

env = gym.make('testenv-v0')
param_space_size = 2
state_space_size = 2
env_param_space_size = 2
episode_length = 2

env_param = EnvParam(env, param_space_size, state_space_size, env_param_space_size, episode_length)

mean_initial_param = np.array([1, -1])
variance_initial_param = 0
variance_action = 0.01
batch_size = 100
num_batch = 100
discount_factor = 1
runs = 20
learning_rate = 1e-1
ess_min = 10
adaptive = "No"

simulation_param = SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size, num_batch, discount_factor, runs, learning_rate, ess_min, adaptive)

policy_params = np.array([[0.5, 0.5], [0.1, 0.1]])
env_params = np.array([[0.9, 1]])

episodes_per_configuration = 1
n_config_cv = policy_params.shape[0] * env_params.shape[0] - 1

estimators = ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"]
results = {}
for estimator in estimators:
    results[estimator] = []

for _ in range(runs):
    # TODO generiamo nuove traiettorie source per ogni run
    [source_task, source_param, episodes_per_config, next_states_unclipped, actions_clipped,
     next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(episode_length, episodes_per_configuration,
                                                                  discount_factor, variance_action, policy_params,
                                                                  env_params, param_space_size, state_space_size,
                                                                  env_param_space_size)

    for estimator in estimators:
        print(estimator)
        off_policy = 0 if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"] else 1
        # TODO Resettiamo il source dataset
        source_dataset = SourceDataset(source_task, source_param, episodes_per_config, next_states_unclipped,
                                       actions_clipped, next_states_unclipped_denoised)
        result = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=off_policy)
        results[estimator].append(result.disc_rewards)

x = range(num_batch)

from scipy.stats import t
alpha = t.interval(0.95, runs-1, loc=0, scale=1)[1] if runs > 1 else 1

means = [np.mean(results[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(results[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Return", names=estimators)
