from joblib import Parallel,delayed
import numpy as np
import datetime
import pickle
import os
import learningAlgorithm as la
import sourceTaskCreation as stc
import simulationClasses as sc
from model_estimation_rkhs import ModelEstimatorRKHS
from discreteModelEstimation import Models
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
    batch_size = 10
    discount_factor = 0.99
    ess_min = 20
    adaptive = "No"
    n_min = 10

    simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size,
                                          num_batch, discount_factor, None, None, ess_min, adaptive, n_min)

    # source task for lqg1d
    episodes_per_configuration = 20

    # source task for lqg1d
    source_dataset_batch_size = 2
    discount_factor = 0.99

    pis = [[-0.1], [-0.2], [-0.3], [-0.4], [-0.5], [-0.6], [-0.7], [-0.8]]
    envs = [[0.9, 1, 0.09], [1.2, 1, 0.09], [1.5, 1, 0.09]]

    policy_params = []
    env_params = []

    for p in pis:
        for e in envs:
            policy_params.append(p)
            env_params.append(e)

    policy_params = np.array(policy_params)
    env_params = np.array(env_params)

    source_envs = []
    for param in np.array(envs):
        source_envs.append(gym.make('LQG1D-v0'))
        source_envs[-1].setParams(param)

    n_config_cv = policy_params.shape[0]
    n_source = [episodes_per_configuration*len(pis) for _ in envs]

    learning_rates = [1e-5, 1e-5, 1e-5, 1e-5, 1e-5]

    [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped,
     next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, episodes_per_configuration,
                                                                  discount_factor, variance_action, policy_params,
                                                                  env_params, param_space_size, state_space_size,
                                                                  env_param_space_size)

    # Proposal envs
    omega_env = np.array([1, 0.9, 1.1, 1.2, 1.05])#np.linspace(env_param_min, env_param_max, linspace_env)
    print(omega_env)
    model_proposals = []

    for i in range(omega_env.shape[0]):
        env_proposal = gym.make('LQG1D-v0')
        env_proposal.setParams(np.concatenate(([omega_env[i]], np.ravel(env_proposal.B), [env_proposal.sigma_noise**2])))
        model_proposals.append(env_proposal)

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
            simulation_param.adaptive = "No"
            dicrete_estimation = None

        else:
            off_policy = 1
            if estimator.endswith("ID"):
                model_estimation = 0
                dicrete_estimation = None

            else:
                if estimator.endswith("DI"):
                    model_estimation = 1
                    dicrete_estimation = 1
                    model = Models(model_proposals)

                else:
                    model_estimation = 1
                    dicrete_estimation = 0
                    model = ModelEstimatorRKHS(kernel_rho=1, kernel_lambda=[1, 1], sigma_env=env_tgt.sigma_noise,
                                               sigma_pi=np.sqrt(variance_action), T=episode_length, R=50, lambda_=0.00,
                                               source_envs=source_envs, n_source=n_source, max_gp=10*5*20, state_dim=1,
                                               linear_kernel=True)
                    if estimator.endswith("GP"):
                        model.use_gp = True

            name = estimator[:-3]
            simulation_param.adaptive = adaptive

        simulation_param.learning_rate = learning_rate
        source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration, next_states_unclipped,
                                          actions_clipped, next_states_unclipped_denoised, n_config_cv)

        result = la.learnPolicy(env_param, simulation_param, source_dataset, name, off_policy=off_policy,
                                model_estimation=model_estimation, dicrete_estimation=dicrete_estimation, model_estimator=model)

        stats[estimator].append(result)

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
n_jobs = 1
# Number of runs
n_runs = 20

estimators = ["PD-MIS-CV-BASELINE-ID", "PD-MIS-CV-BASELINE-ES", "PD-MIS-CV-BASELINE-GP", "PD-MIS-CV-BASELINE-DI", "GPOMDP"]
num_batch = 400

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
