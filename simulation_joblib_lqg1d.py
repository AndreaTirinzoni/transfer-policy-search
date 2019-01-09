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

def simulationParallel(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param): #lqg simulation

    [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationAllCombinations(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size)
    i_learning_rate = 0

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

        stats[estimator].append(result)

        i_learning_rate += 1

    return stats

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

estimators = ["MIS", "MIS-CV-BASELINE", "PD-MIS", "PD-MIS-CV-BASELINE", "GPOMDP"]

learning_rates = [5e-6, 7e-6, 7e-6, 9e-6, 5e-6]
stats = {}
for estimator in estimators:
    stats[estimator] = []

seeds = [np.random.randint(1000000) for _ in range(runs)]

#results = Parallel(n_jobs=16)(delayed(simulationParallel)(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param) for seed in seeds) #cartpole
results = Parallel(n_jobs=8)(delayed(simulationParallel)(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param) for seed in seeds) #lqg1d


with open('results.pkl', 'wb') as output:
    pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
