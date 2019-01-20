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
    friction = np.random.uniform(0.065, 0.196, arguments.n_source_models)
    hole_size = np.random.uniform(0.10, 0.15, arguments.n_source_models)
    envs = [[putter_length[i], friction[i], hole_size[i], 0.09] for i in range(arguments.n_source_models)]

    policy_params = []
    env_params = []

    for p in pis:
        for e in envs:
            policy_params.append(p)
            env_params.append(e)

    policy_params = np.array(policy_params)
    env_params = np.array(env_params)

    n_config_cv = policy_params.shape[0]

    data = stc.sourceTaskCreationSpec(env_src, episode_length, arguments.n_source_samples, arguments.gamma, variance_action,
                                      policy_params, env_params, param_space_size, state_space_size, env_param_space_size,
                                      features=feats, env_target=env_tgt)

    stats = {}
    for estimator in estimators:
        stats[estimator] = []

    for estimator in estimators:

        print(estimator)

        # Create a new dataset object
        source_dataset = sc.SourceDataset(*data, n_config_cv)

        off_policy = 0 if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"] else 1

        name = estimator

        if estimator.endswith("SR"):
            # Create a fake dataset for the sample-reuse algorithm
            data_sr = stc.sourceTaskCreationSpec(env_src, episode_length, 1, arguments.gamma, variance_action,
                                                 np.array([[0, 0, 0, 0]]), np.array([[1.0, 0.131, 0.1, 0.09]]),
                                                 param_space_size, state_space_size, env_param_space_size, features=feats,
                                                 env_target=env_tgt)
            source_dataset = sc.SourceDataset(*data_sr, 1)
            name = estimator[:-3]

        result = la.learnPolicy(env_param, simulation_param, source_dataset, name, off_policy=off_policy,
                                model_estimation=0, dicrete_estimation=0,
                                model_estimator=None, verbose=not arguments.quiet, features=feats)

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
parser.add_argument("--iterations", default=300, type=int)
parser.add_argument("--learning_rate", default=1e-2, type=float)
parser.add_argument("--gamma", default=1.0, type=float)
parser.add_argument("--batch_size", default=10, type=int)
parser.add_argument("--ess_min", default=20, type=int)
parser.add_argument("--n_min", default=5, type=int)
parser.add_argument("--adaptive", default=False, action='store_true')
parser.add_argument("--use_adam", default=False, action='store_true')
parser.add_argument("--n_source_samples", default=10, type=int)
parser.add_argument("--n_source_models", default=5, type=int)
parser.add_argument("--n_jobs", default=1, type=int)
parser.add_argument("--n_runs", default=1, type=int)
parser.add_argument("--quiet", default=False, action='store_true')

# Read arguments
arguments = parser.parse_args()

estimators = ["GPOMDP", "PD-IS", "MIS", "MIS-CV-BASELINE", "PD-MIS", "PD-MIS-CV-BASELINE", "PD-MIS-CV-BASELINE-SR"]

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

with open('{0}/results.pkl'.format(folder), 'wb') as output:
    pickle.dump(results, output)

################################################

print(folder)
