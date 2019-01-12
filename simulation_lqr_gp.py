from joblib import Parallel,delayed
import numpy as np
import datetime
import pickle
import os
import learningAlgorithm as la
import sourceTaskCreation as stc
import simulationClasses as sc
from model_estimation_rkhs import ModelEstimatorRKHS
import gym


def main():

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
    discount_factor = 0.99
    ess_min = 50
    adaptive = "No"

    simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size,
                                          num_batch, discount_factor, None, None, ess_min, adaptive)

    # source task for lqg1d
    episodes_per_configuration = 10

    policy_params = np.array([[-0.1], [-0.2], [-0.3], [-0.4], [-0.5], [-0.6], [-0.7]])
    env_params = np.array([[0.8, 1, 0.09], [1.2, 1, 0.09]])

    source_envs = []
    for param in env_params:
        source_envs.append(gym.make('LQG1D-v0'))
        source_envs[-1].setParams(param)
    n_config_cv = policy_params.shape[0] * env_params.shape[0]
    n_source = [policy_params.shape[0]*episodes_per_configuration for _ in env_params]

    learning_rates = [5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6]

    [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped,
     next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, episodes_per_configuration,
                                                                  discount_factor, variance_action, policy_params,
                                                                  env_params, param_space_size, state_space_size,
                                                                  env_param_space_size)

    stats = {}
    for estimator in estimators:
        stats[estimator] = []

    for estimator,learning_rate in zip(estimators, learning_rates):

        model = None

        print(estimator)

        if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"]:
            off_policy = 0
            model_estimation = 0
            name = estimator
        else:
            off_policy = 1
            if estimator.endswith("ID"):
                model_estimation = 0
            else:
                model_estimation = 1
                model = ModelEstimatorRKHS(kernel_rho=1, kernel_lambda=[1, 1], sigma_env=env_tgt.sigma_noise,
                                           sigma_pi=np.sqrt(variance_action), T=episode_length, R=50, lambda_=0.00,
                                           source_envs=source_envs, n_source=n_source, state_dim=1)
                if estimator.endswith("GP"):
                    model.use_gp = True
            name = estimator[:-3]

        simulation_param.learning_rate = learning_rate
        source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration, next_states_unclipped,
                                          actions_clipped, next_states_unclipped_denoised, n_config_cv)

        result = la.learnPolicy(env_param, simulation_param, source_dataset, name, off_policy=off_policy,
                                model_estimation=model_estimation, dicrete_estimation=0, model_estimator=model)

        stats[estimator].append(result.policy_parameter[:, 0])

    return stats


def run(id, seed):

    # Set the random seed
    np.random.seed(seed)

    print("Starting run {0}".format(id))

    results = main()

    print("Done run {0}".format(id))

    # Log the results
    with open("{0}/{1}.pkl".format(folder, id), 'wb') as output:
        pickle.dump(results, output)

    return results


# Number of jobs
n_jobs = 10
# Number of runs
n_runs = 20

estimators = ["GPOMDP", "MIS-ID", "MIS-CV-BASELINE-ID", "MIS-ES", "MIS-GP", "MIS-CV-BASELINE-ES", "MIS-CV-BASELINE-GP"]
num_batch = 200

# Base folder where to log
folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.mkdir(folder)

# Seeds for each run
seeds = [np.random.randint(1000000) for _ in range(n_runs)]

if n_jobs == 1:
    results = [run(id, seed) for id, seed in zip(range(n_runs), seeds)]
else:
    results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(run)(id, seed) for id, seed in zip(range(n_runs), seeds))

################################################

res = {}
for estimator in estimators:
    res[estimator] = []
for stat in results:
    for estimator in estimators:
        res[estimator].append(stat[estimator])
for estimator in estimators:
    res[estimator] = np.array(res[estimator]).reshape(n_runs, num_batch)

x = range(num_batch)

from scipy.stats import t
alpha = t.interval(0.95, n_runs-1, loc=0, scale=1)[1] if n_runs > 1 else 1

means = [np.mean(res[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(res[estimator], axis=0) / np.sqrt(n_runs) for estimator in estimators]

import utils.plot as plot
plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Return", names=estimators, file_name="plot")
