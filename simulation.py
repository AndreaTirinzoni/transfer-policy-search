import gym
import envs
import numpy as np
import algorithmPolicySearch as alg
import learningAlgorithm as la
import sourceTaskCreation as stc
import pickle
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
        self.mask_weights = None

# # LQG1D
# env = gym.make('LQG1D-v0')
# param_space_size = 1
# state_space_size = 1
# env_param_space_size = 3
# episode_length = 20

env_tgt = gym.make('cartpolec-v0')
env_src = gym.make('cartpolec-v0')
param_space_size = 4
state_space_size = 4
env_param_space_size = 3
episode_length = 200

env_param = EnvParam(env_tgt, param_space_size, state_space_size, env_param_space_size, episode_length)

mean_initial_param = -0.1 * np.ones(param_space_size)
variance_initial_param = 0
variance_action = 0.1
batch_size = 5
num_batch = 120
discount_factor = 0.99
runs = 15
learning_rate = 1e-5
ess_min = 50
adaptive = "No"

simulation_param = SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size, num_batch, discount_factor, runs, learning_rate, ess_min, adaptive)

# source task for lqg1d
# source_dataset_batch_size = 20
# discount_factor = 0.99
# env_param_min = 0.9
# env_param_max = 1
# policy_param_min = -1
# policy_param_max = -0.1
# linspace_env = 2
# linspace_policy = 2
# n_config_cv = (linspace_policy * linspace_env) - 1 #number of configurations to use to fit the control variates
#np.random.seed(2000)

# source task for cartpole
policy_params = np.array([[-0.045, 0.20, 0.24, 0.6], [-0.05, 0.1, 0.1, 0.4], [-0.06, 0.21, 0.24, 0.73], [-0.08, -0.05, 0.05, 0.35]])
env_params = np.array([[1, 0.5, 0.09], [1, 0.5, 0.09], [0.5, 1, 0.09], [0.5, 1, 0.09]])

# policy_params = np.array([[-0.045, 0.20, 0.24, 0.6], [-0.05, 0.1, 0.1, 0.4], [-0.06, 0.21, 0.24, 0.73], [-0.08, -0.05, 0.05, 0.35], [-0.09, 0.16, 0.36, 0.7], [-0.11, -0.17, 0.007, 0.15]])
# env_params = np.array([[1, 0.5, 0.09], [1, 0.5, 0.09], [0.5, 1, 0.09], [0.5, 1, 0.09], [1.5, 1, 0.09], [1.5, 1, 0.09]])

episodes_per_configuration = 20
n_config_cv = policy_params.shape[0] * env_params.shape[0] - 1
# [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, episodes_per_configuration, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size)

# source_task = np.genfromtxt('source_task.csv', delimiter=',')
# episodes_per_config = np.genfromtxt('episodes_per_config.csv', delimiter=',').astype(int)
# source_param = np.genfromtxt('source_param.csv', delimiter=',')
# next_states_unclipped = np.genfromtxt('next_states_unclipped.csv', delimiter=',')
# actions_clipped = np.genfromtxt('actions_clipped.csv', delimiter=',')

#estimators = ["MIS", "MIS-CV", "MIS-CV-BASELINE", "REINFORCE-BASELINE"]
estimators = ["MIS", "MIS-CV-BASELINE", "PD-MIS", "PD-MIS-CV-BASELINE", "GPOMDP"]
#learning_rates = [2e-5, 6e-6, 1e-5, 2e-5, 1e-5, 1e-6, 1e-5, 1e-5, 1e-6, 1e-6, 1e-5]
learning_rates = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
disc_rewards = {}
policy = {}
gradient = {}
ess = {}
n_def = {}
for estimator in estimators:
    disc_rewards[estimator] = []
    policy[estimator] = []
    gradient[estimator] = []
    ess[estimator] = []
    n_def[estimator] = []

for i_run in range(runs):

    # [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationAllCombinations(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size)
    [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, episodes_per_configuration, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size)
    source_dataset = SourceDataset(source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised)

    print("Run: " + str(i_run))
    initial_param = np.random.normal(simulation_param.mean_initial_param, simulation_param.variance_initial_param)

    i_learning_rate = 0
    #[source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationAllCombinations(env_src, episode_length, episodes_per_configuration, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size)

    # off_policy_is = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #1e-6
    # off_policy_pd_is = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #1e-6
    # off_policy_mis = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #1e-5
    # off_policy_mis_cv = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #5e-6
    # off_policy_mis_cv_baseline = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #1e-5
    # off_policy_pd_mis = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1) #1e-6
    # off_policy_pd_mis_cv_baseline_approx = la.learnPolicy(env_param, simulation_param, source_dataset, estimator) #8e-6
    # off_policy_pd_mis_cv_baseline = la.learnPolicy(env_param, simulation_param, source_dataset, estimator) #1e-5
    # reinforce = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=0) #1e-6
    # reinforce_baseline = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=0) #1e-6
    # gpomdp = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=0) #1e-6

    for estimator in estimators:

        print(estimator)
        if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"]:
            off_policy = 0
            simulation_param.batch_size = 10
        else:
            off_policy = 1

        source_dataset = SourceDataset(source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised)
        simulation_param.learning_rate = learning_rates[i_learning_rate]

        result = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=off_policy) #1e-6

        disc_rewards[estimator].append(result.disc_rewards)
        policy[estimator].append(result.policy_parameter)
        gradient[estimator].append(result.gradient)
        ess[estimator].append(result.ess)
        n_def[estimator].append(result.n_def)

        i_learning_rate += 1

with open('rewards.pkl', 'wb') as output:
    pickle.dump(disc_rewards, output, pickle.HIGHEST_PROTOCOL)

with open('policy.pkl', 'wb') as output:
    pickle.dump(policy, output, pickle.HIGHEST_PROTOCOL)

with open('gradient.pkl', 'wb') as output:
    pickle.dump(gradient, output, pickle.HIGHEST_PROTOCOL)

with open('ess.pkl', 'wb') as output:
    pickle.dump(ess, output, pickle.HIGHEST_PROTOCOL)

with open('n_def.pkl', 'wb') as output:
    pickle.dump(n_def, output, pickle.HIGHEST_PROTOCOL)

# with open('rewards.pkl', 'rb') as input:
#     rewards = pickle.load(input)
# x = range(num_batch)
#
# mean_alg1 = np.mean(disc_mis, axis=0)
# mean_alg2 = np.mean(disc_mis_cv, axis=0)
# mean_alg3 = np.mean(disc_mis_cv_baseline, axis=0)
# mean_alg4 = np.mean(disc_reinforce_baseline, axis=0)
# var_alg1 = np.std(disc_mis, axis=0) / (m.sqrt(runs))
# var_alg2 = np.std(disc_mis_cv, axis=0) / (m.sqrt(runs))
# var_alg3 = np.std(disc_mis_cv_baseline, axis=0) / (m.sqrt(runs))
# var_alg4 = np.std(disc_reinforce_baseline, axis=0) / (m.sqrt(runs))
#
# plot.plot_curves([x, x, x, x], [mean_alg1, mean_alg2, mean_alg3, mean_alg4], [var_alg1, var_alg2, var_alg3, var_alg4], title="Discounted rewards over batches", x_label="Batch", y_label="Disc reward", names=["MIS", "MIS-CV", "MIS-CV-BASELINE", "REINFORCE-BASELINE"])
#
# mean_pol1 = np.mean(policy_mis, axis=0)
# mean_pol2 = np.mean(policy_mis_cv, axis=0)
# mean_pol3 = np.mean(policy_mis_cv_baseline, axis=0)
# mean_pol4 = np.mean(policy_reinforce_baseline, axis=0)
# var_pol1 = np.std(policy_mis, axis=0) / (m.sqrt(runs))
# var_pol2 = np.std(policy_mis_cv, axis=0) / (m.sqrt(runs))
# var_pol3 = np.std(policy_mis_cv_baseline, axis=0) / (m.sqrt(runs))
# var_pol4 = np.std(policy_reinforce_baseline, axis=0) / (m.sqrt(runs))
#
# for i in range(param_space_size):
#     plot.plot_curves([x, x, x, x], [mean_pol1[:, i], mean_pol2[:, i], mean_pol3[:, i], mean_pol4[:, i]], [var_pol1[:, i], var_pol2[:, i], var_pol3[:, i], var_pol4[:, i]], title="Policy param over batches", x_label="Batch", y_label="Policy param", names=["MIS", "MIS-CV", "MIS-CV-BASELINE", "REINFORCE-BASELINE"])
#     #print("optimal: " + str(mean_pol3[-1, i]) + " middle: " + str(mean_pol3[100, i]))
