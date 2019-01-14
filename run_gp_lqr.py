import gym
import numpy as np
import learningAlgorithm as la
import sourceTaskCreation as stc
import simulationClasses as sc
from model_estimation_rkhs import ModelEstimatorRKHS


def simulationParallel(env_src, episode_length, discount_factor, variance_action, episodes_per_configuration, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param, rkhs_model, seed):

    np.random.seed(seed)

    [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped,
     next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, episodes_per_configuration,
                                                                  discount_factor, variance_action, policy_params,
                                                                  env_params, param_space_size, state_space_size,
                                                                  env_param_space_size)
    i_learning_rate = 0

    stats = {}
    for estimator in estimators:
        stats[estimator] = []

    for estimator in estimators:

        print(estimator)
        if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"]:
            off_policy = 0
            model_estimation = 0
        else:
            off_policy = 1
            model_estimation = 1

        simulation_param.learning_rate = learning_rates[i_learning_rate]
        source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised, n_config_cv)

        result = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=off_policy, model_estimation=model_estimation, dicrete_estimation=0, model_estimator=rkhs_model)

        stats[estimator].append(result.policy_parameter[:,0])

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
num_batch = 10
discount_factor = 0.99
runs = 5
learning_rate = 1e-8
ess_min = 50
adaptive = "No"

simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size, num_batch, discount_factor, runs, learning_rate, ess_min, adaptive)

# source task for lqg1d
episodes_per_configuration = 10

policy_params = np.array([[-0.1],[-0.2],[-0.3],[-0.4],[-0.5],[-0.6],[-0.7]])
env_params = np.array([[1, 1, 0.09],[1, 1, 0.09]])

source_envs = []
for param in env_params:
    source_envs.append(gym.make('LQG1D-v0'))
    source_envs[-1].setParams(param)
n_config_cv = policy_params.shape[0] * env_params.shape[0]

estimators = ["MIS"]

learning_rates = [1e-5, 1e-6]

seeds = [np.random.randint(1000000) for _ in range(runs)]

model = ModelEstimatorRKHS(kernel_rho=1, kernel_lambda=[1, 1], sigma_env=env_tgt.sigma_noise, sigma_pi=np.sqrt(variance_action), T=episode_length, R=50, lambda_=0.00, source_envs=source_envs, state_dim=1)
model.use_gp = True

results_stats = [simulationParallel(env_src, episode_length, discount_factor, variance_action, episodes_per_configuration, param_space_size, state_space_size, env_param_space_size, estimators, learning_rates, env_param, simulation_param, model, seed) for seed in seeds]
results = {}
for estimator in estimators:
    results[estimator] = []
for stat in results_stats:
    for estimator in estimators:
        results[estimator].append(stat[estimator])
for estimator in estimators:
    results[estimator] = np.array(results[estimator]).reshape(runs,num_batch)

x = range(num_batch)

from scipy.stats import t
alpha = t.interval(0.95, runs-1, loc=0, scale=1)[1] if runs > 1 else 1

means = [np.mean(results[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(results[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

import utils.plot as plot
plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Return", names=estimators, file_name="plot")