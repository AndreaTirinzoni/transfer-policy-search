import gym
import envs
import numpy as np
import algorithmPolicySearch as alg
import learningAlgorithm as la
import sourceTaskCreation as stc
from utils import plot
import math as m

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

# # LQG1D
# env = gym.make('LQG1D-v0')
# param_space_size = 1
# state_space_size = 1
# env_param_space_size = 3
# episode_length = 20

env = gym.make('LQG1D-v0')
param_space_size = 1
state_space_size = 1
env_param_space_size = 3
episode_length = 20

env_param = EnvParam(env, param_space_size, state_space_size, env_param_space_size, episode_length)

mean_initial_param = -0.1 * np.ones(param_space_size)
variance_initial_param = 0
variance_action = 0.1
batch_size = 20
num_batch = 400
discount_factor = 0.99
runs = 4
learning_rate = 1e-5
ess_min = 100
adaptive = "Yes"

simulation_param = SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size, num_batch, discount_factor, runs, learning_rate, ess_min, adaptive)

# source task for lqg1d
episodes_per_configuration = 2
discount_factor = 0.99
env_param_min = 0.9
env_param_max = 1.1
policy_param_min = -1
policy_param_max = -0.1
linspace_env = 3
linspace_policy = 2
n_config_cv = (linspace_policy * linspace_env) - 1 #number of configurations to use to fit the control variates

[source_task, source_param, episodes_per_config, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationAllCombinations(env, episode_length, episodes_per_configuration, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size)

# source task for cartpole
# policy_params = np.array([[-0.045, 0.20, 0.24, 0.6], [-0.05, 0.1, 0.1, 0.4]])
# env_params = np.array([[1, 0.5, 0.09], [1, 0.5, 0.09]])
#
# # policy_params = np.array([[-0.045, 0.20, 0.24, 0.6], [-0.05, 0.1, 0.1, 0.4], [-0.06, 0.21, 0.24, 0.73], [-0.08, -0.05, 0.05, 0.35], [-0.09, 0.16, 0.36, 0.7], [-0.11, -0.17, 0.007, 0.15]])
# # env_params = np.array([[1, 0.5, 0.09], [1, 0.5, 0.09], [0.5, 1, 0.09], [0.5, 1, 0.09], [1.5, 1, 0.09], [1.5, 1, 0.09]])
#
# episodes_per_configuration = 1
# n_config_cv = policy_params.shape[0] * env_params.shape[0] - 1
# [source_task, source_param, episodes_per_config, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env, episode_length, episodes_per_configuration, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size)

# source_task = np.genfromtxt('source_task.csv', delimiter=',')
# episodes_per_config = np.genfromtxt('episodes_per_config.csv', delimiter=',').astype(int)
# source_param = np.genfromtxt('source_param.csv', delimiter=',')
# next_states_unclipped = np.genfromtxt('next_states_unclipped.csv', delimiter=',')
# actions_clipped = np.genfromtxt('actions_clipped.csv', delimiter=',')

source_dataset = SourceDataset(source_task, source_param, episodes_per_config, next_states_unclipped, actions_clipped, next_states_unclipped_denoised)

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
    print("MIS-CV-BASELINE")
    estimator = "MIS-CV-BASELINE"
    off_policy_mis_cv_baseline = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #1e-6

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

    # print("REINFORCE")
    # estimator = "REINFORCE"
    # reinforce = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=0) #1e-6
    #
    # print("REINFORCE-BASELINE")
    # estimator = "REINFORCE-BASELINE"
    # reinforce_baseline = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=0) #1e-6

    print("GPOMDP")
    estimator = "GPOMDP"
    gpomdp = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=0) #1e-6

x = range(num_batch)

#mean_pol1 = np.mean(off_policy_mis_cv_baseline.disc_rewards, axis=0)
mean_pol3 = np.mean(gpomdp.disc_rewards, axis=0)
#var_pol1 = np.std(off_policy_mis_cv_baseline.disc_rewards, axis=0) / (m.sqrt(runs))
var_pol3 = np.std(gpomdp.disc_rewards, axis=0) / (m.sqrt(runs))

plot.plot_curves([x, x], [mean_pol1, mean_pol3], [var_pol1, var_pol3], title="Discounted rewards over batches", x_label="Batch", y_label="Rewards", names=["MIS-CV-BASELINE", "G(PO)MDP"])

mean_pol1 = np.mean(off_policy_mis_cv_baseline.policy_parameter, axis=0)
mean_pol3 = np.mean(gpomdp.policy_parameter, axis=0)
var_pol1 = np.std(off_policy_mis_cv_baseline.policy_parameter, axis=0) / (m.sqrt(runs))
var_pol3 = np.std(gpomdp.policy_parameter, axis=0) / (m.sqrt(runs))


for i in range(param_space_size):
    plot.plot_curves([x, x], [mean_pol1[:, i], mean_pol3[:, i]], [var_pol1[:, i], var_pol3[:, i]], title = "Policy parameter over batches", x_label = "Batch", y_label = "Policy parameter", names = ["MIS-CV-BASELINE", "G(PO)MDP"])
    #print("optimal: " + str(mean_pol3[-1, i]) + " middle: " + str(mean_pol3[100, i]))