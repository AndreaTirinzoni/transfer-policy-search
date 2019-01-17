from joblib import Parallel,delayed
import numpy as np
import datetime
import pickle
import os
import learningAlgorithm as la
import sourceTaskCreation as stc
import simulationClasses as sc
import gym


def main():

    env_tgt = gym.make('minigolf-v1')
    env_src = gym.make('minigolf-v1')
    param_space_size = 6
    state_space_size = 6
    env_param_space_size = 4
    episode_length = 20

    env_param = sc.EnvParam(env_tgt, param_space_size, state_space_size, env_param_space_size, episode_length)

    mean_initial_param = np.random.normal(np.ones(param_space_size) * 0.8, 0.001)
    mean_initial_param = np.array([1,1.1,1.2,1.3,1.4,1.5])
    variance_initial_param = 0
    variance_action = 0.1
    batch_size = 10
    discount_factor = 1

    simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size,
                                          num_batch, discount_factor, None, None, None, "No", 0)

    source_dataset_batch_size = 10
    n_config_cv = 1
    policy_params = np.array([[-0.5]])
    env_params = np.array([[1.0, 1.0, 0.09]])

    [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped,
     next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, source_dataset_batch_size,
                                                                  discount_factor, variance_action, policy_params, env_params,
                                                                  param_space_size, state_space_size, env_param_space_size)

    stats = {}
    for estimator in estimators:
        stats[estimator] = []

    for estimator,learning_rate in zip(estimators, learning_rates):

        print(estimator)

        if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"]:
            off_policy = 0
        else:
            off_policy = 1

        source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration,
                                          next_states_unclipped,
                                          actions_clipped, next_states_unclipped_denoised, n_config_cv)

        simulation_param.learning_rate = learning_rate

        result = la.learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=off_policy)

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
n_jobs = 4

# Number of runs
n_runs = 20

estimators = ["GPOMDP"]
learning_rates = [1e-3, 1e-3, 1e-3, 1e-3]
num_batch = 300

# Base folder where to log
folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.mkdir(folder)

# Seeds for each run
seeds = [np.random.randint(1000000) for _ in range(n_runs)]

if n_jobs == 1:
    results = [run(id, seed) for id, seed in zip(range(n_runs), seeds)]
else:
    results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(run)(id, seed) for id, seed in zip(range(n_runs), seeds))

print(folder)

res = {}
for estimator in estimators:
    res[estimator] = []
for stat in results:
    for estimator in estimators:
        res[estimator].append(stat[estimator][0].disc_rewards)
for estimator in estimators:
    res[estimator] = np.array(res[estimator]).reshape(n_runs, num_batch)

x = range(num_batch)

from scipy.stats import t
alpha = t.interval(0.95, n_runs-1, loc=0, scale=1)[1] if n_runs > 1 else 1

means = [np.mean(res[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(res[estimator], axis=0) / np.sqrt(n_runs) for estimator in estimators]

import utils.plot as plot
plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Disc", names=estimators, file_name="plot")

###

res = {}
for estimator in estimators:
    res[estimator] = []
for stat in results:
    for estimator in estimators:
        res[estimator].append(stat[estimator][0].total_rewards)
for estimator in estimators:
    res[estimator] = np.array(res[estimator]).reshape(n_runs, num_batch)

x = range(num_batch)

from scipy.stats import t
alpha = t.interval(0.95, n_runs-1, loc=0, scale=1)[1] if n_runs > 1 else 1

means = [np.mean(res[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(res[estimator], axis=0) / np.sqrt(n_runs) for estimator in estimators]

import utils.plot as plot
plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Total", names=estimators, file_name="plot")

param = np.array([1,1.1,1.2,1.3,1.4,1.5])
env = gym.make('minigolf-v1')

state = env.reset()
true_s = env.get_true_state()
for t in range(20):
    action = np.dot(param, state) + np.random.randn() * 0.1
    state, reward, done,_, _, _ = env.step(action)
    true_ns = env.get_true_state()
    print([true_s,action,true_ns,reward])
    true_s = true_ns
    if done:
        break