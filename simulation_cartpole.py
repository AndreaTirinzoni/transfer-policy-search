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

    env_tgt = gym.make('cartpolec-v0')
    env_src = gym.make('cartpolec-v0')
    param_space_size = 4
    state_space_size = 4
    env_param_space_size = 3
    episode_length = 200

    env_param = sc.EnvParam(env_tgt, param_space_size, state_space_size, env_param_space_size, episode_length)

    mean_initial_param = np.random.normal(np.zeros(param_space_size), np.sqrt(0.1))
    variance_initial_param = 0
    variance_action = 0.1
    batch_size = 5
    discount_factor = 0.99
    ess_min = 20
    adaptive = "Yes"
    n_min = 10

    simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size,
                                          num_batch, discount_factor, None, None, ess_min, adaptive, n_min)


    # source task for cartpole
    #policy_params = np.array([[-0.0259, 0.2541, 0.2797, 0.7054], [-0.0659, 0.0610, 0.0910, 0.3631]])#[-0.049, 0.176, 0.447, 0.810], [-0.105, -0.16, -0.0103, 0.128], [-0.131, 0.246, 0.402, 0.854], [-0.103, 0.158, -0.023, 0.124], [-0.039, 0.299, 0.386, 0.782], [-0.103, -0.137, -0.038, 0.12], [-0.111, -0.148, -0.027, 0.086], [-0.0115, 0.219, 0.416, 0.792], [-0.049, 0.176, 0.447, 0.810], [-0.105, -0.16, -0.0103, 0.128]])
    policy_params = np.array([[-0.0781, 0.1737, 0.2883, 0.6518], [-0.0757, -0.0379, 0.0585, 0.3026], [-0.0554, 0.1725, 0.2940, 0.6916], [-0.0796, -0.0363, 0.0526, 0.3152], [-0.103, 0.158, -0.023, 0.124], [-0.039, 0.299, 0.386, 0.782]])#, [-0.103, -0.137, -0.038, 0.12], [-0.111, -0.148, -0.027, 0.086]])#, , [-0.0115, 0.219, 0.416, 0.792], [-0.049, 0.176, 0.447, 0.810], [-0.105, -0.16, -0.0103, 0.128]])
    env_params = np.array([[1.1, 0.6, 0.09], [1.1, 0.6, 0.09], [1.5, 0.5, 0.09], [1.5, 0.5, 0.09], [0.8, 0.8, 0.09], [0.8, 0.8, 0.09]])#, [1.5, 0.5, 0.09], [1.5, 0.5, 0.09]])#, , [1.2, 0.9, 0.09], [1.2, 0.9, 0.09], ])

    source_dataset_batch_size = 20
    n_config_cv = policy_params.shape[0]

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
            name = estimator
            simulation_param.batch_size = 15
        else:
            off_policy = 1
            simulation_param.batch_size = 5

        simulation_param.learning_rate = learning_rate
        if estimator.endswith("SR"): #if sample reuse
            source_dataset_batch_size = 1
            simulation_param.batch_size = 5
            discount_factor = 0.99
            policy_params = np.array([[-0.5, 0.8, 0.9, 1]])
            env_params = np.array([[0.8, 0.8, 0.09]])
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
n_jobs = 20

# Number of runs
n_runs = 20

estimators = ["PD-IS", "PD-MIS-CV-BASELINE", "PD-MIS-CV-BASELINE-SR", "GPOMDP"]
#estimators = ["PD-MIS-SR", "PD-MIS-CV-BASELINE-SR", "GPOMDP"]
learning_rates = [9e-4, 9e-4, 9e-4, 9e-4]
num_batch = 70

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
