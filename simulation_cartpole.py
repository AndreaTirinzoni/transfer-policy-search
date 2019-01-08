import gym
import envs
import numpy as np
import algorithmPolicySearch as alg
import learningAlgorithm as la
import sourceTaskCreation as stc
import pickle
from utils import plot
import math as m
import simulationClasses as sc

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
num_batch = 200
discount_factor = 0.99
runs = 5
learning_rate = 1e-5
ess_min = 50
adaptive = "No"

simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size, num_batch, discount_factor, runs, learning_rate, ess_min, adaptive)

# source task for cartpole
policy_params = np.array([[-0.045, 0.20, 0.24, 0.6], [-0.05, 0.1, 0.1, 0.4]])
env_params = np.array([[1, 0.5, 0.09], [1, 0.5, 0.09]])

# policy_params = np.array([[-0.13, 0.19, 0.28, 0.57], [-0.04, 0.1, 0.11, 0.37], [-0.01, 0.09, 0.26, 0.61], [-0.03, 0.01, 0.1, 0.32], [-0.09, 0.11, 0.36, 0.67], [-0.1, -0.11, 0.04, 0.19]])
# env_params = np.array([[1, 0.5, 0.09], [1, 0.5, 0.09], [0.5, 1, 0.09], [0.5, 1, 0.09], [1.5, 1, 0.09], [1.5, 1, 0.09]])

number_of_episodes_per_config = 15
n_config_cv = policy_params.shape[0] * env_params.shape[0]

estimators = ["PD-MIS-CV-BASELINE", "PD-MIS-CV-BASELINE-APPROXIMATED", "GPOMDP"]

#learning_rates = [2e-5, 6e-6, 1e-5, 2e-5, 1e-5, 1e-6, 1e-5, 1e-5, 1e-6, 1e-6, 1e-5]
learning_rates = [1e-4, 1e-4]#, 2e-3, 1e-4]
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

    [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, number_of_episodes_per_config, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size)
    source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised, n_config_cv)

    print("Run: " + str(i_run))
    initial_param = np.random.normal(simulation_param.mean_initial_param, simulation_param.variance_initial_param)

    i_learning_rate = 0

    for estimator in estimators:

        print(estimator)
        if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"]:
            off_policy = 0
            simulation_param.batch_size = 20
        else:
            off_policy = 1

        source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised, n_config_cv)
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
