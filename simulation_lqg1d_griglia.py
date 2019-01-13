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


    source_dataset_batch_size = 20
    discount_factor = 0.99
    env_param_min = 0.5
    env_param_max = 1.5
    policy_param_min = -1
    policy_param_max = -0.1
    linspace_env = 11
    linspace_policy = 10
    n_config_cv = linspace_policy * linspace_env

    [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped,
     next_states_unclipped_denoised] = stc.sourceTaskCreationAllCombinations(env_src, episode_length, source_dataset_batch_size,
                                                                             discount_factor, variance_action, env_param_min,
                                                                             env_param_max, policy_param_min, policy_param_max,
                                                                             linspace_env, linspace_policy, param_space_size,
                                                                             state_space_size, env_param_space_size)

    stats = {}
    for estimator in estimators:
        stats[estimator] = []

    for estimator,learning_rate in zip(estimators, learning_rates):

        print(estimator)

        if estimator in ["GPOMDP", "REINFORCE", "REINFORCE-BASELINE"]:
            off_policy = 0
            name = estimator
            simulation_param.batch_size = 10
        else:
            off_policy = 1

        simulation_param.learning_rate = learning_rate
        if estimator.endswith("SR"): #if sample reuse
            source_dataset_batch_size = 1
            discount_factor = 0.99
            policy_params = np.array([[-1]])
            env_params = np.array([[1-5, 1, 0.09]])
            n_config_cv = 1
            name = estimator[:-3]
            [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped,
             next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, source_dataset_batch_size,
                                                                          discount_factor, variance_action, policy_params,
                                                                          env_params, param_space_size, state_space_size,
                                                                          env_param_space_size)
        else:
            name = estimator
        source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration, next_states_unclipped,
                                          actions_clipped, next_states_unclipped_denoised, n_config_cv)

        simulation_param.learning_rate = learning_rate

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
n_jobs = 10

# Number of runs
n_runs = 20

estimators = ["IS", "PD-IS", "MIS", "MIS-CV-BASELINE", "PD-MIS", "PD-MIS-CV-BASELINE", "GPOMDP"]
learning_rates = [1e-6, 2e-6, 5e-6, 7e-6, 7e-6, 1e-5, 1e-5]
num_batch = 420

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
################################################

# res = {}
# for estimator in estimators:
#     res[estimator] = []
# for stat in results:
#     for estimator in estimators:
#         res[estimator].append(stat[estimator])
# for estimator in estimators:
#     res[estimator] = np.array(res[estimator]).reshape(n_runs, num_batch)
