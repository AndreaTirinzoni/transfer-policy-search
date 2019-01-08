import gym
import envs
import numpy as np
import algorithmPolicySearch as alg
import learningAlgorithm as la
import sourceTaskCreation as stc
import pickle
from utils import plot
import math as m
from joblib import Parallel
from joblib import delayed
import simulationClasses as sc

#def simulationParallel(env_src, episode_length, episodes_per_configuration, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param): #cartpolec simulation
def simulationParallel(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param): #lqg simulation

    [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationAllCombinations(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size)
    # [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, episodes_per_configuration, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size)
    i_learning_rate = 0
    #[source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationAllCombinations(env_src, episode_length, episodes_per_configuration, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size)

    for estimator in estimators:

        print(estimator)
        if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"]:
            off_policy = 0
            simulation_param.batch_size = 15
        else:
            off_policy = 1

        source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised)
        simulation_param.learning_rate = learning_rates[i_learning_rate]

        result = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=off_policy)

        disc_rewards[estimator].append(result.disc_rewards)
        policy[estimator].append(result.policy_parameter)
        gradient[estimator].append(result.gradient)
        ess[estimator].append(result.ess)
        n_def[estimator].append(result.n_def)

        i_learning_rate += 1

    return [disc_rewards, policy, gradient, ess, n_def]

env_tgt = gym.make('LQG1D-v0')
env_src = gym.make('LQG1D-v0')
param_space_size = 1
state_space_size = 1
env_param_space_size = 3
episode_length = 20

env_param = sc.EnvParam(env_tgt, param_space_size, state_space_size, env_param_space_size, episode_length)

mean_initial_param = -0.1 * np.ones(param_space_size)
variance_initial_param = 0
variance_action = 0.1
batch_size = 5
num_batch = 400
discount_factor = 0.99
runs = 16
learning_rate = 1e-5
ess_min = 70
adaptive = "No"

simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size, num_batch, discount_factor, runs, learning_rate, ess_min, adaptive)

# source task for lqg1d
source_dataset_batch_size = 4
discount_factor = 0.99
env_param_min = 0.9
env_param_max = 1
policy_param_min = -1
policy_param_max = -0.1
linspace_env = 2
linspace_policy = 2
n_config_cv = (linspace_policy * linspace_env) - 1 #number of configurations to use to fit the control variates
#np.random.seed(2000)

# source task for cartpole
# policy_params = np.array([[-0.045, 0.20, 0.24, 0.6], [-0.05, 0.1, 0.1, 0.4], [-0.06, 0.21, 0.24, 0.73], [-0.08, -0.05, 0.05, 0.35]])
# env_params = np.array([[1, 0.5, 0.09], [1, 0.5, 0.09], [0.5, 1, 0.09], [0.5, 1, 0.09]])

# policy_params = np.array([[-0.045, 0.20, 0.24, 0.6], [-0.05, 0.1, 0.1, 0.4], [-0.06, 0.21, 0.24, 0.73], [-0.08, -0.05, 0.05, 0.35], [-0.09, 0.16, 0.36, 0.7], [-0.11, -0.17, 0.007, 0.15]])
# env_params = np.array([[1, 0.5, 0.09], [1, 0.5, 0.09], [0.5, 1, 0.09], [0.5, 1, 0.09], [1.5, 1, 0.09], [1.5, 1, 0.09]])

# episodes_per_configuration = 20
# n_config_cv = policy_params.shape[0] * env_params.shape[0] - 1
# [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, episodes_per_configuration, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size)

estimators = ["MIS", "MIS-CV-BASELINE", "PD-MIS", "PD-MIS-CV-BASELINE", "GPOMDP"]

#learning_rates = [2e-5, 6e-6, 1e-5, 2e-5, 1e-5, 1e-6, 1e-5, 1e-5, 1e-6, 1e-6, 1e-5]
learning_rates = [1e-5, 1e-5, 2e-5, 5e-6, 1e-5]
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

seeds = [np.random.randint(1000000) for _ in range(runs)]

#results = Parallel(n_jobs=16)(delayed(simulationParallel)(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param) for seed in seeds) #cartpole
results = Parallel(n_jobs=8)(delayed(simulationParallel)(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param) for seed in seeds) #lqg1d


with open('results.pkl', 'wb') as output:
    pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
