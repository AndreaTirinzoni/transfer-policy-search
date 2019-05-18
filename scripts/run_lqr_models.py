import sys
sys.path.append("../")

import argparse
from joblib import Parallel,delayed
import numpy as np
import datetime
import pickle
import os
import learning_algorithm as la
import source_task_creation as stc
import simulation_classes as sc
from model_estimation_rkhs import ModelEstimatorRKHS
from discrete_model_estimation import Models
import gym


def main():

    # General env properties
    env_tgt = gym.make('LQG1D-v0')
    env_src = gym.make('LQG1D-v0')
    param_space_size = 1
    state_space_size = 1
    env_param_space_size = 3
    episode_length = 20
    gaussian_transitions = True

    env_param = sc.EnvParam(env_tgt, param_space_size, state_space_size, env_param_space_size, episode_length, gaussian_transitions)

    mean_initial_param = -0.1 * np.ones(param_space_size)
    variance_initial_param = 0
    variance_action = 0.1

    simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, arguments.batch_size,
                                          arguments.iterations, arguments.gamma, None, arguments.learning_rate, arguments.ess_min,
                                          "No", arguments.n_min, use_adam=arguments.use_adam)

    # Source tasks
    pis = [[-0.1], [-0.2], [-0.3], [-0.4], [-0.5], [-0.6], [-0.7], [-0.8]]
    A = np.random.uniform(0.6, 1.4, arguments.n_source_models)
    B = np.random.uniform(0.8, 1.2, arguments.n_source_models)
    envs = [[A[i], B[i], 0.09] for i in range(A.shape[0])]

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
    n_source = [arguments.n_source_samples*len(pis) for _ in envs]

    data = stc.sourceTaskCreationSpec(env_src, episode_length, arguments.n_source_samples, arguments.gamma, variance_action,
                                      policy_params, env_params, param_space_size, state_space_size, env_param_space_size)

    # Envs for discrete model estimation
    possible_env_params = [[1.0, 1.0, 0.09],
                           [1.5, 1.0, 0.09],
                           [0.5, 1.0, 0.09],
                           [1.2, 0.8, 0.09],
                           [0.8, 1.2, 0.09],
                           [1.1, 0.9, 0.09],
                           [0.9, 1.1, 0.09],
                           [1.5, 0.5, 0.09]]

    possible_envs = []
    for param in np.array(possible_env_params):
        possible_envs.append(gym.make('LQG1D-v0'))
        possible_envs[-1].setParams(param)

    stats = {}
    for estimator in estimators:
        stats[estimator] = []

    for estimator in estimators:

        print(estimator)

        model_estimation = 0
        off_policy = 0
        discrete_estimation = 0
        model = None

        # Create a new dataset object
        source_dataset = sc.SourceDataset(*data, n_config_cv)

        if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"]:
            name = estimator
        else:
            off_policy = 1
            name = estimator[:-3]

            if estimator.endswith("SR"):
                # Create a fake dataset for the sample-reuse algorithm
                data_sr = stc.sourceTaskCreationSpec(env_src, episode_length, 1, arguments.gamma, variance_action,
                                                  np.array([[-0.1]]), np.array([[1.0, 1.0, 0.09]]), param_space_size,
                                                  state_space_size, env_param_space_size)
                source_dataset = sc.SourceDataset(*data_sr, 1)
            elif estimator.endswith("DI"):
                model_estimation = 1
                discrete_estimation = 1
                model = Models(possible_envs)
            elif estimator.endswith("GP") or estimator.endswith("ES") or estimator.endswith("MI"):
                model_estimation = 1
                model = ModelEstimatorRKHS(kernel_rho=1, kernel_lambda=[1, 1], sigma_env=env_tgt.sigma_noise,
                                           sigma_pi=np.sqrt(variance_action), T=episode_length, R=arguments.rkhs_samples,
                                           lambda_=0.0, source_envs=source_envs, n_source=n_source,
                                           max_gp=arguments.max_gp_samples, state_dim=1, linear_kernel=True,
                                           balance_coeff=arguments.balance_coeff,
                                           target_env=env_tgt if arguments.print_mse else None)
                if estimator.endswith("GP"):
                    model.use_gp = True
                elif estimator.endswith("MI"):
                    model.use_gp_generate_mixture = True

        result = la.learnPolicy(env_param, simulation_param, source_dataset, name, off_policy=off_policy,
                                model_estimation=model_estimation, dicrete_estimation=discrete_estimation,
                                model_estimator=model, verbose=not arguments.quiet)

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


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--iterations", default=100, type=int)
parser.add_argument("--learning_rate", default=1e-2, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--batch_size", default=10, type=int)
parser.add_argument("--ess_min", default=20, type=int)
parser.add_argument("--n_min", default=1, type=int)
parser.add_argument("--adaptive", default=False, action='store_true')
parser.add_argument("--use_adam", default=False, action='store_true')
parser.add_argument("--n_source_samples", default=10, type=int)
parser.add_argument("--n_source_models", default=5, type=int)
parser.add_argument("--max_gp_samples", default=1000, type=int)
parser.add_argument("--rkhs_samples", default=50, type=int)
parser.add_argument("--dump_model", default=False, action='store_true')
parser.add_argument("--dump_model_iteration", default=10, type=int)
parser.add_argument("--balance_coeff", default=False, action='store_true')
parser.add_argument("--print_mse", default=False, action='store_true')
parser.add_argument("--n_jobs", default=1, type=int)
parser.add_argument("--n_runs", default=1, type=int)
parser.add_argument("--quiet", default=False, action='store_true')

# Read arguments
arguments = parser.parse_args()

estimators = ["PD-MIS-CV-BASELINE-DI", "GPOMDP",
              "PD-MIS-CV-BASELINE-SR",
              "PD-MIS-CV-BASELINE-ID",
              "PD-MIS-CV-BASELINE-ES",
              "PD-MIS-CV-BASELINE-GP",
              "PD-MIS-CV-BASELINE-DI"]

# Base folder where to log
folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.mkdir(folder)

# Save arguments
with open("{0}/params.txt".format(folder), 'w') as f:
    for key, value in vars(arguments).items():
        f.write("{0}: {1}\n".format(key, value))

# Seeds for each run
seeds = [np.random.randint(1000000) for _ in range(arguments.n_runs)]

if arguments.n_jobs == 1:
    results = [run(id, seed) for id, seed in zip(range(arguments.n_runs), seeds)]
else:
    results = Parallel(n_jobs=arguments.n_jobs, backend='loky')(delayed(run)(id, seed) for id, seed in zip(range(arguments.n_runs), seeds))

with open('{0}/results1.pkl'.format(folder), 'wb') as output:
    pickle.dump(results, output)

################################################

print(folder)
