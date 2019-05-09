import sys
sys.path.append("../")

import argparse
from joblib import Parallel,delayed
import numpy as np
import datetime
import pickle
import os
import learningAlgorithm_noGaussianTransitions as la
import sourceTaskCreation as stc
import simulationClasses as sc
from model_estimation_rkhs import ModelEstimatorRKHS
from discreteModelEstimation import Models
from source_estimator import SourceEstimator
import gym
from features import polynomial


def main():

    # General env properties
    env_tgt = gym.make('minigolf-v0')
    env_src = gym.make('minigolf-v0')
    param_space_size = 4
    state_space_size = 1
    env_param_space_size = 4
    episode_length = 20

    env_param = sc.EnvParam(env_tgt, param_space_size, state_space_size, env_param_space_size, episode_length)

    mean_initial_param = np.random.normal(np.ones(param_space_size) * 0.2, 0.01)
    variance_initial_param = 0
    variance_action = 0.1
    feats = polynomial

    simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, arguments.batch_size,
                                          arguments.iterations, arguments.gamma, None, arguments.learning_rate, arguments.ess_min,
                                          "Yes" if arguments.adaptive else "No", arguments.n_min, use_adam=arguments.use_adam)

    # Source tasks
    pis = [[0.20097172, 0.20182519, 0.19957835, 0.20096946],
           [0.34099334, 0.21422279, 0.20053974, 0.20105477],
           [0.46923638, 0.22986188, 0.20266549, 0.20137892],
           [0.64977232, 0.26575410, 0.21014003, 0.20300604],
           [0.89955698, 0.32707635, 0.23490234, 0.21518798],
           [1.09006747, 0.35577241, 0.24517702, 0.22017502],
           [1.22329955, 0.40621784, 0.28787368, 0.24836521],
           [1.34824502, 0.43750823, 0.29981691, 0.25448715],
           [1.24846429, 0.42882867, 0.27008977, 0.22433061],
           [1.41946655, 0.53908188, 0.33195278, 0.25586648]]

    putter_length = np.random.uniform(0.7, 1.0, arguments.n_source_models)
    friction = np.random.uniform(0.1, 0.15, arguments.n_source_models)
    hole_size = np.random.uniform(0.10, 0.15, arguments.n_source_models)
    envs = [[putter_length[i], friction[i], hole_size[i], 0.09] for i in range(arguments.n_source_models)]

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
        source_envs.append(gym.make('minigolf-v0'))
        source_envs[-1].setParams(param)
        source_envs[-1].min_variance = 1e-1
    n_config_cv = policy_params.shape[0]
    n_source = [arguments.n_source_samples*len(pis) for _ in envs]

    data = stc.sourceTaskCreationSpec(env_src, episode_length, arguments.n_source_samples, arguments.gamma, variance_action,
                                      policy_params, env_params, param_space_size, state_space_size, env_param_space_size,
                                      features=feats, env_target=env_tgt)

    # Envs for discrete model estimation
    possible_env_params = envs  # TODO possible envs are the source envs

    possible_envs = []
    for param in np.array(possible_env_params):
        possible_envs.append(gym.make('minigolf-v0'))
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
                                                     np.array([[0, 0, 0, 0]]), np.array([[1.0, 0.131, 0.1, 0.09]]),
                                                     param_space_size, state_space_size, env_param_space_size,
                                                     features=feats, env_target=env_tgt)
                source_dataset = sc.SourceDataset(*data_sr, 1)
            elif estimator.endswith("DI"):
                model_estimation = 1
                discrete_estimation = 1
                model = Models(possible_envs)
            elif estimator.endswith("GP") or estimator.endswith("ES") or estimator.endswith("MI") or estimator.endswith("NS"):
                model_estimation = 1
                model = ModelEstimatorRKHS(kernel_rho=10, kernel_lambda=[100, 10], sigma_env=env_tgt.sigma_noise,
                                           sigma_pi=np.sqrt(variance_action), T=arguments.rkhs_horizon, R=arguments.rkhs_samples,
                                           lambda_=0.0, source_envs=source_envs, n_source=n_source,
                                           max_gp=arguments.max_gp_samples, state_dim=1, linear_kernel=False,
                                           balance_coeff=arguments.balance_coeff, alpha_gp=1,
                                           print_mse=arguments.print_mse, features=polynomial,
                                           param_dim=param_space_size, target_env=env_tgt, heteroscedastic=True)
                if estimator.endswith("GP"):# or estimator.endswith("NS"):
                    model.use_gp = True
                elif estimator.endswith("MI"):
                    model.use_gp_generate_mixture = True
                if estimator.endswith("NS"):
                    n_models = int(source_dataset.episodes_per_config.shape[0]/source_dataset.policy_per_model)
                    transition_models = []
                    for i in range(n_models):
                        model_estimator = ModelEstimatorRKHS(kernel_rho=10, kernel_lambda=[100, 10], sigma_env=env_tgt.sigma_noise,
                                           sigma_pi=np.sqrt(variance_action), T=arguments.rkhs_horizon, R=arguments.rkhs_samples,
                                           lambda_=0.0, source_envs=source_envs, n_source=n_source,
                                           max_gp=arguments.max_gp_samples, state_dim=1, linear_kernel=False,
                                           balance_coeff=arguments.balance_coeff, alpha_gp=1,
                                           print_mse=arguments.print_mse, features=polynomial,
                                           param_dim=param_space_size, target_env=source_envs[i], heteroscedastic=True, max_gp_src=arguments.max_gp_samples_src)
                        transition_models.append(model_estimator)
                    env_src_models = SourceEstimator(source_dataset, transition_models)
        result = la.learnPolicy(env_param, simulation_param, source_dataset, name, off_policy=off_policy,
                                model_estimation=model_estimation, dicrete_estimation=discrete_estimation,
                                model_estimator=model, verbose=not arguments.quiet, features=polynomial,
                                source_estimator=env_src_models if estimator.endswith("NS") else None)

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
parser.add_argument("--gamma", default=1.0, type=float)
parser.add_argument("--batch_size", default=10, type=int)
parser.add_argument("--ess_min", default=20, type=int)
parser.add_argument("--n_min", default=5, type=int)
parser.add_argument("--adaptive", default=False, action='store_true')
parser.add_argument("--use_adam", default=False, action='store_true')
parser.add_argument("--n_source_samples", default=10, type=int)
parser.add_argument("--n_source_models", default=5, type=int)
parser.add_argument("--max_gp_samples", default=1000, type=int)
parser.add_argument("--max_gp_samples_src", default=4000, type=int)
parser.add_argument("--rkhs_samples", default=20, type=int)
parser.add_argument("--rkhs_horizon", default=20, type=int)
parser.add_argument("--balance_coeff", default=False, action='store_true')
parser.add_argument("--print_mse", default=False, action='store_true')
parser.add_argument("--n_jobs", default=1, type=int)
parser.add_argument("--n_runs", default=1, type=int)
parser.add_argument("--quiet", default=False, action='store_true')

# Read arguments
arguments = parser.parse_args()

estimators = ["MIS-CV-BASELINE-NS"]

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
