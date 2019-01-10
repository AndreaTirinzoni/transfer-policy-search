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

def simulationParallel(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param, seed): #lqg simulation

    np.random.seed(seed)

    [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size)

    i_learning_rate = 0

    stats = {}
    for estimator in estimators:
        stats[estimator] = []

    for estimator in estimators:

        print(estimator)
        if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"]:
            off_policy = 0
            simulation_param.batch_size = 10
        else:
            off_policy = 1
            simulation_param.batch_size = 10

        source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised, n_config_cv)
        simulation_param.learning_rate = learning_rates[i_learning_rate]

        result = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=off_policy, model_estimation=0, multid_approx=0)

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
batch_size = 10
num_batch = 350
discount_factor = 0.99
runs = 20
learning_rate = 1e-5
ess_min = 70
adaptive = "No"

simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size, num_batch, discount_factor, runs, learning_rate, ess_min, adaptive)

# source task for lqg1d
# source task for lqg1d
source_dataset_batch_size = 1
discount_factor = 0.99
policy_params = np.array([[-1]])
env_params = np.array([[1-5, 1, 0.09]])

n_config_cv = policy_params.shape[0] * env_params.shape[0]

estimators = ["IS", "PD-IS", "MIS", "MIS-CV-BASELINE", "PD-MIS", "PD-MIS-CV-BASELINE", "GPOMDP"]

learning_rates = [1e-6, 2e-6, 5e-6, 7e-6, 7e-6, 1e-5, 1e-5]#, 7e-6, 7e-6, 9e-6, 5e-6]

seeds = [np.random.randint(1000000) for _ in range(runs)]

#results = Parallel(n_jobs=3)(delayed(simulationParallel)(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param, seed) for seed in seeds) #lqg1d
results = [simulationParallel(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param, seed) for seed in seeds]

with open('results.pkl', 'wb') as output:
    pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
