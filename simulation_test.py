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
runs = 10
learning_rate = 1e-1
ess_min = 10
adaptive = "No"

simulation_param = SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size, num_batch, discount_factor, runs, learning_rate, ess_min, adaptive)

policy_params = np.array([[0.5, 0.5], [0.1, 0.1]])
env_params = np.array([[0.9, 0.09]])

episodes_per_configuration = 1
n_config_cv = policy_params.shape[0] * env_params.shape[0] - 1
[source_task, source_param, episodes_per_config, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env, episode_length, episodes_per_configuration, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size)

source_dataset = SourceDataset(source_task, source_param, episodes_per_config, next_states_unclipped, actions_clipped, next_states_unclipped_denoised)

rews = []

for i_run in range(runs):

    # print("Run: " + str(i_run))
    # initial_param = np.random.normal(simulation_param.mean_initial_param, simulation_param.variance_initial_param)
    #
    # #[source_task, source_param, episodes_per_config, next_states_unclipped, actions_clipped] = stc.sourceTaskCreation(episode_length, episodes_per_configuration, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max)
    #
    # print("IS")
    # estimator = "IS"
    # off_policy_is = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #1e-6

    # print("PD-IS")
    # estimator = "PD-IS"
    # off_policy_pd_is = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #1e-6
    #
    # print("MIS")
    # estimator = "MIS"
    # off_policy_mis = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #1e-6

    # print("MIS-CV")
    # estimator = "MIS-CV"
    # off_policy_mis_cv = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #1e-6
    #
    #print("MIS-CV-BASELINE")
    #estimator = "MIS-CV-BASELINE"
    #off_policy_mis_cv_baseline = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #1e-6

    # print("PD-MIS")
    # estimator = "PD-MIS"
    # off_policy_pd_mis = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #1e-6
    #
    # print("PD-MIS-CV")
    # estimator = "PD-MIS-CV"
    # off_policy_pd_mis_cv = la.learnPolicy(env_param, simulation_param, source_dataset, estimator) #1e-6
    #
    # print("PD-MIS-CV-BASELINE-APPROXIMATED")
    # estimator = "PD-MIS-CV-BASELINE-APPROXIMATED"
    # off_policy_pd_mis_cv_baseline_approx = la.learnPolicy(env_param, simulation_param, source_dataset, estimator) #1e-6
    # #
    # print("PD-MIS-CV-BASELINE")
    # estimator = "PD-MIS-CV-BASELINE"
    # off_policy_pd_mis_cv_baseline = la.learnPolicy(env_param, simulation_param, source_dataset, estimator) #1e-6

    #print("REINFORCE")
    #estimator = "REINFORCE"
    #reinforce = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=0) #1e-6
    #
    print("REINFORCE-BASELINE")
    estimator = "REINFORCE-BASELINE"
    reinforce_baseline = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=0) #1e-6
    rews.append(reinforce_baseline.disc_rewards)

    #print("GPOMDP")
    #estimator = "GPOMDP"
    #gpomdp = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=0) #1e-6

x = range(num_batch)

rews = np.array(rews)
mean1 = np.mean(rews, axis=0)
var1 = np.std(rews, axis=0) / (np.sqrt(runs))

plot.plot_curves([x], [mean1], [var1], x_label="Iteration", y_label="Return", names=["Reinforce"])
print(reinforce_baseline.policy_parameter)
