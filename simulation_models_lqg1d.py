import gym
import envs
import numpy as np
import learningAlgorithm as la
import sourceTaskCreation as stc
import pickle
from utils import plot
import math as m
import simulationClasses as sc

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
batch_size = 1
num_batch = 120
discount_factor = 0.99
runs = 10
learning_rate = 1e-5
ess_min = 50
adaptive = "No"

simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size, num_batch, discount_factor, runs, learning_rate, ess_min, adaptive)

# source task for lqg1d
source_dataset_batch_size = 20
discount_factor = 0.99
env_param_min = 0.8
env_param_max = 1.3
policy_param_min = -1
policy_param_max = -0.1
linspace_env = 6
linspace_policy = 2
n_config_cv = linspace_policy * linspace_env #number of configurations to use to fit the control variates

# Proposal envs
omega_env = np.array([0.8, 0.9, 1, 1.1, 1.2, 1.01, 1.001])#np.linspace(env_param_min, env_param_max, linspace_env)
print(omega_env)
model_proposals = []

for i in range(omega_env.shape[0]):
    env_proposal = gym.make('LQG1D-v0')
    env_proposal.setParams(np.concatenate(([omega_env[i]], np.ravel(env_proposal.B), [env_proposal.sigma_noise**2])))
    model_proposals.append(env_proposal)

estimators = ["MIS-CV-BASELINE", "PD-MIS-MIS-BASELINE-APPROXIMATED", "GPOMDP"]

#learning_rates = [2e-5, 6e-6, 1e-5, 2e-5, 1e-5, 1e-6, 1e-5, 1e-5, 1e-6, 1e-6, 1e-5]
learning_rates = [2e-3, 2e-3, 1e-4]
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

    [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationAllCombinations(env_src, episode_length, source_dataset_batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size)
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

        result = la.learnPolicyWithModelEstimation(env_param, simulation_param, source_dataset, model_proposals, estimator, off_policy=off_policy) #1e-6

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
