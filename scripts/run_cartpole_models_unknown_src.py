import sys
sys.path.append("../")

import argparse
from joblib import Parallel,delayed
import numpy as np
import datetime
import pickle
import os
import learning_algorithm_unknown_src as la
import source_task_creation as stc
import simulation_classes as sc
from model_estimation_rkhs import ModelEstimatorRKHS
from source_estimator import SourceEstimator
from discrete_model_estimation import Models
import gym


def main(id):

    # General env properties
    env_tgt = gym.make('cartpolec-v0')
    env_src = gym.make('cartpolec-v0')
    param_space_size = 4
    state_space_size = 4
    env_param_space_size = 3
    episode_length = 200

    env_param = sc.EnvParam(env_tgt, param_space_size, state_space_size, env_param_space_size, episode_length)

    mean_initial_param = np.random.normal(np.zeros(param_space_size), 0.01)
    variance_initial_param = 0
    variance_action = 0.1

    simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, arguments.batch_size,
                                          arguments.iterations, arguments.gamma, None, arguments.learning_rate, arguments.ess_min,
                                          "Yes" if arguments.adaptive else "No", arguments.n_min, use_adam=arguments.use_adam)

    # Source tasks
    pis = [[-0.04058811, 0.06820783, 0.09962419, -0.01481458],
           [-0.04327763, 0.01926409, 0.10651812, 0.07304843],
           [-0.04660533, -0.08301117, 0.14598312, 0.31524803],
           [-0.04488895, -0.04959011, 0.20856307, 0.52564195],
           [-0.02085553, 0.11530108, 0.24525215, 0.58338479],
           [-0.03072567, 0.15546779, 0.27241488, 0.65833969],
           [-0.05493752, 0.11100809, 0.30213226, 0.73134919],
           [-0.02389198, 0.18004238, 0.30697023, 0.72447482],
           [-0.0702051, 0.17653729, 0.32254312, 0.72004621],
           [-0.09675066, 0.16063462, 0.32343255, 0.73801456]]

    m = np.random.uniform(0.8, 1.2, arguments.n_source_models)
    l = np.random.uniform(0.4, 0.6, arguments.n_source_models)
    envs = [[m[i], l[i], 0.09] for i in range(m.shape[0])]

    policy_params = []
    env_params = []
    num_policy = len(pis)
    for e in envs:
        for p in pis:
            policy_params.append(p)
            env_params.append(e)

    policy_params = np.array(policy_params)
    env_params = np.array(env_params)

    source_envs = []
    for param in np.array(envs):
        source_envs.append(gym.make('cartpolec-v0'))
        source_envs[-1].setParams(param)
    n_config_cv = policy_params.shape[0]
    n_source = [arguments.n_source_samples*len(pis) for _ in envs]

    data = stc.sourceTaskCreationSpec(env_src, episode_length, arguments.n_source_samples, arguments.gamma, variance_action,
                                      policy_params, env_params, param_space_size, state_space_size, env_param_space_size)

    # Envs for discrete model estimation
    possible_env_params = [[1.0, 0.5, 0.09],
                           [0.8, 0.3, 0.09],
                           [1.2, 0.7, 0.09],
                           [1.1, 0.6, 0.09],
                           [0.9, 0.4, 0.09],
                           [0.9, 0.6, 0.09],
                           [1.1, 0.4, 0.09],
                           [1.5, 1.0, 0.09]]

    possible_envs = []
    for param in np.array(possible_env_params):
        possible_envs.append(gym.make('cartpolec-v0'))
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
        env_src_models = None

        # Create a new dataset object
        source_dataset = sc.SourceDataset(*data, n_config_cv)
        source_dataset.policy_per_model = num_policy
        if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"]:
            name = estimator
        else:
            off_policy = 1
            name = estimator[:-3]

            if estimator.endswith("SR"):
                # Create a fake dataset for the sample-reuse algorithm
                data_sr = stc.sourceTaskCreationSpec(env_src, episode_length, 1, arguments.gamma, variance_action,
                                                     np.array([[0, 0, 0, 0]]), np.array([[1.0, 0.5, 0.09]]), param_space_size,
                                                     state_space_size, env_param_space_size)
                source_dataset = sc.SourceDataset(*data_sr, 1)
            elif estimator.endswith("DI"):
                model_estimation = 1
                discrete_estimation = 1
                model = Models(possible_envs)
            elif estimator.endswith("GP") or estimator.endswith("ES") or estimator.endswith("MI") or estimator.endswith("NS"):
                model_estimation = 1
                model = ModelEstimatorRKHS(kernel_rho=1, kernel_lambda=[1, 1, 1, 1, 1], sigma_env=env_tgt.sigma_env,
                                           sigma_pi=np.sqrt(variance_action), T=arguments.rkhs_horizon, R=arguments.rkhs_samples,
                                           lambda_=0.0, source_envs=source_envs, n_source=n_source,
                                           max_gp=arguments.max_gp_samples, state_dim=4, linear_kernel=False,
                                           balance_coeff=arguments.balance_coeff, alpha_gp=1e-5,
                                           target_env=env_tgt if arguments.print_mse else None, id=id)
                if estimator.endswith("GP"):
                    model.use_gp = True
                elif estimator.endswith("MI"):
                    model.use_gp_generate_mixture = True

                if estimator.endswith("NS"):
                    n_models = int(source_dataset.episodes_per_config.shape[0]/source_dataset.policy_per_model)
                    transition_models = []
                    for i in range(n_models):
                        model_estimator = ModelEstimatorRKHS(kernel_rho=1, kernel_lambda=[1, 1, 1, 1, 1], sigma_env=env_tgt.sigma_env,
                                               sigma_pi=np.sqrt(variance_action), T=arguments.rkhs_horizon, R=arguments.rkhs_samples,
                                               lambda_=0.0, source_envs=source_envs, n_source=n_source,
                                               max_gp=arguments.max_gp_samples_src, state_dim=4, linear_kernel=False,
                                               balance_coeff=arguments.balance_coeff, alpha_gp=1e-5,
                                               target_env=env_tgt if arguments.print_mse else None, id=id)
                        transition_models.append(model_estimator)
                    env_src_models = SourceEstimator(source_dataset, transition_models)
        result = la.learnPolicy(env_param, simulation_param, source_dataset, name, off_policy=off_policy,
                                model_estimation=model_estimation, dicrete_estimation=discrete_estimation,
                                model_estimator=model, verbose=not arguments.quiet, dump_model=arguments.dump_estimated_model,
                                iteration_dump=arguments.iteration_dump, source_estimator=env_src_models if estimator.endswith("NS") else None)

        stats[estimator].append(result)

    return stats


def run(id, seed):

    # Set the random seed
    np.random.seed(seed)

    print("Starting run {0}".format(id))

    results = main(id)

    print("Done run {0}".format(id))

    # Log the results
    with open("{0}/{1}.pkl".format(folder, id), 'wb') as output:
        pickle.dump(results, output)

    return results


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--iterations", default=50, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--batch_size", default=10, type=int)
parser.add_argument("--ess_min", default=20, type=int)
parser.add_argument("--n_min", default=3, type=int)
parser.add_argument("--adaptive", default=False, action='store_true')
parser.add_argument("--use_adam", default=False, action='store_true')
parser.add_argument("--n_source_samples", default=10, type=int)
parser.add_argument("--n_source_models", default=2, type=int)
parser.add_argument("--max_gp_samples", default=250, type=int)
parser.add_argument("--max_gp_samples_src", default=1000, type=int)
parser.add_argument("--rkhs_samples", default=20, type=int)
parser.add_argument("--rkhs_horizon", default=20, type=int)
parser.add_argument("--dump_estimated_model", default=False, action='store_true')
parser.add_argument("--source_task_unknown", default=False, action='store_true')
parser.add_argument("--iteration_dump", default=5, type=int)
parser.add_argument("--balance_coeff", default=False, action='store_true')
parser.add_argument("--print_mse", default=False, action='store_true')
parser.add_argument("--n_jobs", default=1, type=int)
parser.add_argument("--n_runs", default=6, type=int)
parser.add_argument("--quiet", default=False, action='store_true')

# Read arguments
arguments = parser.parse_args()

estimators = ["GPOMDP",
              "PD-MIS-NS",
              "PD-MIS-SR",
              "PD-MIS-ID",
              "PD-MIS-ES",
              "PD-MIS-GP",
              "PD-MIS-DI"]

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
