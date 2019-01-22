from joblib import Parallel,delayed
import numpy as np
import datetime
import pickle
import os
import learningAlgorithm as la
import simulationClasses as sc
import gym
from envs.planning_env import Planning_env


def main():
    """
    lqg1d sample reuse
    """
    env_tgt = gym.make('LQG1D-v0')

    with open('gp1.pkl', 'rb') as input:
        transition_model = pickle.load(input)

    variance_env = env_tgt.getParam()[-1]

    env_planning = Planning_env(transition_model, env_tgt, np.sqrt(variance_env))
    param_space_size = 1
    state_space_size = 1
    env_param_space_size = 3
    episode_length = 20

    env_param = sc.EnvParam(env_planning, param_space_size, state_space_size, env_param_space_size, episode_length)

    mean_initial_param = -0.1 * np.ones(param_space_size)
    variance_initial_param = 0
    variance_action = 0.1
    batch_size = 10
    discount_factor = 0.99
    ess_min = 25
    adaptive = "No"
    n_min = 5

    simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size,
                                          num_batch, discount_factor, None, None, ess_min, adaptive, n_min)

    stats = {}
    for estimator in estimators:
        stats[estimator] = []

    for estimator in estimators:

        print(estimator)

        off_policy = 0
        name = estimator
        simulation_param.batch_size = 10

        simulation_param.learning_rate = learning_rate

        source_dataset = None



        result = la.learnPolicy(env_param, simulation_param, source_dataset, name, off_policy=off_policy)

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
n_jobs = 2

# Number of runs
n_runs = 6

estimators = ["GPOMDP"]
learning_rate = 1e-5
num_batch = 350

# Base folder where to log
folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.mkdir(folder)

# Seeds for each run
seeds = [np.random.randint(1000000) for _ in range(n_runs)]

if n_jobs == 1:
    results = [run(id, seed) for id, seed in zip(range(n_runs), seeds)]
else:
    results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(run)(id, seed) for id, seed in zip(range(n_runs), seeds))

with open('results.pkl', 'wb') as output:
    pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
