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

def simulationParallel(env_src, episode_length, episodes_per_configuration, discount_factor, variance_action, policy_params, env_params, n_config_cv, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param, seed): #cartpolec simulation

    np.random.seed(seed)

    [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, episodes_per_configuration, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size)
    i_learning_rate = 0

    stats = {}
    for estimator in estimators:
        stats[estimator] = []

    for estimator in estimators:

        print(estimator)
        if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"]:
            off_policy = 0
            simulation_param.batch_size = 15
        else:
            off_policy = 1
            simulation_param.batch_size = 5

        source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised, n_config_cv)
        simulation_param.learning_rate = learning_rates[i_learning_rate]

        result = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=off_policy, model_estimation=0, multid_approx=0)

        stats[estimator].append(result)

        i_learning_rate += 1

    return stats

env_tgt = gym.make('cartpolec-v0')
env_src = gym.make('cartpolec-v0')
param_space_size = 4
state_space_size = 4
env_param_space_size = 3
episode_length = 200

env_param = sc.EnvParam(env_tgt, param_space_size, state_space_size, env_param_space_size, episode_length)

mean_initial_param = -0.1 * np.ones(param_space_size)
variance_initial_param = 0
variance_action = 0.1
batch_size = 5
num_batch = 100
discount_factor = 0.99
runs = 5
learning_rate = 1e-5
ess_min = 50
adaptive = "Yes"

simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size, num_batch, discount_factor, runs, learning_rate, ess_min, adaptive)

# source task for cartpole

policy_params = np.array([[-0.131, 0.246, 0.402, 0.854], [-0.103, 0.158, -0.023, 0.124], [-0.039, 0.299, 0.386, 0.782], [-0.103, -0.137, -0.038, 0.12], [-0.049, 0.176, 0.447, 0.810], [-0.105, -0.16, -0.0103, 0.128], [-0.111, -0.148, -0.027, 0.086], [-0.0115, 0.219, 0.416, 0.792]])
env_params = np.array([[0.8, 0.8, 0.09], [0.8, 0.8, 0.09], [1.5, 0.5, 0.09], [1.5, 0.5, 0.09], [0.5, 1, 0.09], [0.5, 1, 0.09], [1.2, 0.9, 0.09], [1.2, 0.9, 0.09]])

source_dataset_batch_size = 20
n_config_cv = policy_params.shape[0] * env_params.shape[0]

estimators = ["PD-IS", "PD-MIS", "PD-MIS-CV-BASELINE-APPROXIMATED", "GPOMDP"]

learning_rates = [7e-4, 1e-3, 2e-3, 5e-4]

seeds = [np.random.randint(1000000) for _ in range(runs)]

results = Parallel(n_jobs=10)(delayed(simulationParallel)(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, policy_params, env_params, n_config_cv, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param, seed) for seed in seeds) #cartpole
#results = [simulationParallel(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, policy_params, env_params, n_config_cv, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param, seed) for seed in seeds]

with open('results.pkl', 'wb') as output:
    pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
