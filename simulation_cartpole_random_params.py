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

    mean_initial_param = np.random.normal(np.zeros(param_space_size), 0.01)
    variance_initial_param = 0
    variance_action = 0.1
    batch_size = 5
    discount_factor = 0.99
    ess_min = 10
    adaptive = "Yes"
    n_min = 1

    simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size,
                                          num_batch, discount_factor, None, None, ess_min, adaptive, n_min)
    # source task for lqg1d
    source_dataset_batch_size = 10
    discount_factor = 0.99

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

    m = np.random.uniform(0.8, 1.4, 5)
    l = np.random.uniform(0.3, 0.7, 5)
    variance_env = 0.09
    envs = []
    for i in range(len(m)):
        envs.append([m[i], l[i], variance_env])

    policy_params = []
    env_params = []

    for p in pis:
        for e in envs:
            policy_params.append(p)
            env_params.append(e)

    policy_params = np.array(policy_params)
    env_params = np.array(env_params)

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
        else:
            off_policy = 1

        simulation_param.learning_rate = learning_rate
        if estimator.endswith("SR"): #if sample reuse
            source_dataset_batch_size = 1
            policy_params = np.array([[0, 0, 0, 0]])
            env_params = np.array([[1.0, 0.5, 0.09]])
            name = estimator[:-3]
            [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped,
             next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_src, episode_length, source_dataset_batch_size,
                                                                          discount_factor, variance_action, policy_params,
                                                                          env_params, param_space_size, state_space_size,
                                                                          env_param_space_size)

            source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration,
                                              next_states_unclipped,
                                              actions_clipped, next_states_unclipped_denoised, 1)
        else:
            name = estimator

            source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration,
                                              next_states_unclipped,
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

estimators = ["GPOMDP", "PD-IS", "PD-MIS-CV-BASELINE", "PD-MIS-CV-BASELINE-SR"]
learning_rates = [1e-3, 1e-3, 1e-3, 1e-3]
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

print(folder)

################################################

# res = {}
# for estimator in estimators:
#     res[estimator] = []
# for stat in results:
#     for estimator in estimators:
#         res[estimator].append(stat[estimator])
# for estimator in estimators:
#     res[estimator] = np.array(res[estimator]).reshape(n_runs, num_batch)