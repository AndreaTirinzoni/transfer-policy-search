from joblib import Parallel,delayed
import numpy as np
import datetime
import pickle
import os
import learningAlgorithm as la
import simulationClasses as sc
import gym
import sourceTaskCreation as stc
from envs.planning_env import Planning_env
from features import identity


def main(transition_model):
    """
    lqg1d sample reuse
    """
    env_tgt = gym.make('cartpolec-v0')

    variance_env = 0#env_tgt.getEnvParam()[-1]

    env_planning = Planning_env(transition_model, env_tgt, np.sqrt(variance_env))
    param_space_size = 4
    state_space_size = 4
    env_param_space_size = 3
    episode_length = 200

    env_param = sc.EnvParam(env_planning, param_space_size, state_space_size, env_param_space_size, episode_length)

    mean_initial_param = np.random.normal(np.zeros(param_space_size), 0.01)
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

        source_dataset_batch_size = 1
        policy_params = np.array([[0, 0, 0, 0]])
        env_params = np.array([[1.0, 0.5, 0.09]])
        name = estimator[:-3]
        [source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped,
         next_states_unclipped_denoised] = stc.sourceTaskCreationSpec(env_tgt, episode_length, source_dataset_batch_size,
                                                                      discount_factor, variance_action, policy_params,
                                                                      env_params, param_space_size, state_space_size,
                                                                      env_param_space_size)

        source_dataset = sc.SourceDataset(source_task, source_param, episodes_per_configuration,
                                          next_states_unclipped,
                                          actions_clipped, next_states_unclipped_denoised, 1)

        off_policy = 0
        name = estimator
        simulation_param.batch_size = 10

        simulation_param.learning_rate = learning_rate


        result = la.learnPolicy(env_param, simulation_param, source_dataset, name, off_policy=off_policy)

        stats[estimator].append(result)

    return stats


def run(id, seed, model_dump, run_dump):

    # Set the random seed
    np.random.seed(seed)

    dump_id = model_dump*10
    with open('fitted_model_'+str(run_dump) + "_"+str(dump_id)+'.pkl', 'rb') as input:
        transition_model = pickle.load(input)

    print("Starting run {0}".format(id))

    results = main(transition_model)

    print("Done run {0}".format(id))

    # Log the results
    with open("{0}/{1}.pkl".format(folder, id), 'wb') as output:
        pickle.dump(results, output)

    return results


# Number of jobs
n_jobs = 2

# Number of runs
n_runs = 2

estimators = ["GPOMDP"]
learning_rate = 1e-4
num_batch = 20
n_dump = 5
n_run_dump = 6

# Base folder where to log
folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.mkdir(folder)

result_different_dump = []

for run_dump in range(n_run_dump):
    for model_dump in range(n_dump):
    # Seeds for each run
        seeds = [np.random.randint(1000000) for _ in range(n_runs)]

        if n_jobs == 1:
            results = [run(id, seed, model_dump, run_dump) for id, seed in zip(range(n_runs), seeds)]
        else:
            results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(run)(id, seed, model_dump, run_dump) for id, seed in zip(range(n_runs), seeds))

        with open(folder + '/results_' + str(run_dump) + "_" + str(model_dump) + '.pkl', 'wb') as output:
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

policy = []
rewards = []
rewards_dumps = []
tot_rewards_real = []

x = range(num_batch)
#folder = "20190122_165737"
param_space_size = 4
mean_initial_param = np.zeros(param_space_size)
variance_initial_param = np.zeros(param_space_size)
batch_size = 10
discount_factor = 0.99
ess_min = 20
adaptive = "No"
n_min = 3
variance_action = 0.1

simulation_param = sc.SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size,
                                      num_batch, discount_factor, None, None, ess_min, adaptive, n_min)

features = identity

env_tgt = gym.make('cartpolec-v0')

variance_env = env_tgt.getEnvParam()[-1]
param_space_size = 4
state_space_size = 4
env_param_space_size = 3
episode_length = 200
discount_factor_timestep = np.asarray([discount_factor ** i for i in range(episode_length)])

env_param = sc.EnvParam(env_tgt, param_space_size, state_space_size, env_param_space_size, episode_length)

rewards = {}

for model_dump in range(n_dump):
    rewards[model_dump] = []

for model_dump in range(n_dump):
    for run_dump in range(n_run_dump):
        with open(folder + '/results_' + str(run_dump) + "_" + str(model_dump) + '.pkl', 'rb') as input:
            results = pickle.load(input)

        for i in range(n_runs):
            policy_current_run = results[i]["GPOMDP"][0].policy_parameter
            opt_policy = policy_current_run[-1]
            tot_rewards_real_current_gp = la.generateEpisodesAndComputeRewards(env_param, simulation_param, opt_policy, discount_factor_timestep, features)

            rewards[model_dump].append(tot_rewards_real_current_gp)


with open(folder + '/tot_rewards_per_iteration.pkl', 'wb') as output:
        pickle.dump(rewards, output, pickle.HIGHEST_PROTOCOL)
