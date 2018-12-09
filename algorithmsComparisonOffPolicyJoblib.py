import gym
import envs
import numpy as np
import algorithmPolicySearch as alg
import importanceWeights as iw
import math as m
from joblib import Parallel, delayed
import sourceTaskCreation as stc

def simulation(env, batch_size, discount_factor, variance_action, episode_length, mean_initial_param, variance_initial_param, num_batch, learning_rate, seed, episodes_per_configuration, env_param_min, env_param_max, policy_param_min, policy_param_max):
    """
    Function that runs the policy search algorithm for learning the optimal policy
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discount factor
    :param variance_action: variance of the action's distribution
    :param episode_length: length of every episode
    :param mean_initial_param: mean of the initial parameter's distribution
    :param variance_initial_param: variance of the initial parameter's distribution
    :param num_batch: number of batch
    :param learning_rate: learning rate of the update rule
    :param seed: seed of the run
    :param episodes_per_configuration: number of episodes of the source task for every configuration
    :param env_param_min: minimum of the environment variable
    :param env_param_max: maximum of the environment variable
    :param policy_param_min: minimum of the policy parameter variable
    :param policy_param_max: maximum of the policy parameter variable
    :return: A BatchStats object
    """

    np.random.seed(seed)
    initial_param = np.random.normal(mean_initial_param, m.sqrt(variance_initial_param))

    print("Creating source task")
    [source_task, source_param, episodes_per_config, next_states_unclipped, actions_clipped] = stc.sourceTaskCreation(episode_length, episodes_per_configuration, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy)

    print("Learning policy")

    print("IS")
    off_policy_importance_sampling = iw.offPolicyImportanceSampling(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, ess_min, adaptive, learning_rate = 1e-6)

    print("PD-IS")
    off_policy_importance_sampling_pd = iw.offPolicyImportanceSamplingPd(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, ess_min, adaptive, learning_rate = 1e-6)

    print("REINFORCE")
    reinforce = alg.reinforce(env, num_batch, 5, discount_factor, episode_length, initial_param, variance_action, learning_rate)

    print("MIS")
    off_policy_multiple_importance_sampling = iw.offPolicyMultipleImportanceSampling(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, ess_min, adaptive, learning_rate = 1e-7)

    print("MIS-CV")
    off_policy_multiple_importance_sampling_cv = iw.offPolicyMultipleImportanceSamplingCv(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, ess_min, adaptive, n_config_cv, learning_rate = 1e-6)

    print("MIS-CV-BASELINE")
    off_policy_multiple_importance_sampling_cv_baseline = iw.offPolicyMultipleImportanceSamplingCvBaseline(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, ess_min, adaptive, n_config_cv, learning_rate = 1e-5)

    print("PD-MIS")
    off_policy_multiple_importance_sampling_pd = iw.offPolicyMultipleImportanceSamplingPd(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, ess_min, adaptive, learning_rate = 1e-5)

    print("PD-MIS-CV")
    off_policy_multiple_importance_sampling_pd_cv = iw.offPolicyMultipleImportanceSamplingCvPd(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, ess_min, adaptive, n_config_cv, learning_rate = 1e-7)

    print("PD-MIS-CV-BASELINE-APPROXIMATED")
    off_policy_multiple_importance_sampling_pd_cv_baseline_approximated = iw.offPolicyMultipleImportanceSamplingCvPdBaselineApproximated(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, ess_min, adaptive, n_config_cv, learning_rate = 1e-5)

    print("PD-MIS-CV-BASELINE")
    off_policy_multiple_importance_sampling_pd_cv_baseline = iw.offPolicyMultipleImportanceSamplingCvPdBaseline(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, ess_min, adaptive, n_config_cv, learning_rate = 1e-5)

    return [off_policy_importance_sampling, off_policy_importance_sampling_pd, reinforce, off_policy_multiple_importance_sampling,
            off_policy_multiple_importance_sampling_cv, off_policy_multiple_importance_sampling_cv_baseline, off_policy_multiple_importance_sampling_pd,
            off_policy_multiple_importance_sampling_pd_cv, off_policy_multiple_importance_sampling_pd_cv_baseline_approximated, off_policy_multiple_importance_sampling_pd_cv_baseline]


np.set_printoptions(precision=4)
env = gym.make('LQG1D-v0')

mean_initial_param = -0.1
episode_length = 20
variance_initial_param = 0
variance_action = 0.1
batch_size = 1
ess_min = 90
num_batch = 600
discount_factor = 0.99
runs = 10
learning_rate = 1e-6
adaptive = "No"

discounted_reward_off_policy_importance_sampling = np.zeros((runs, num_batch))
discounted_reward_off_policy_importance_sampling_pd = np.zeros((runs, num_batch))
discounted_reward_off_policy_multiple_importance_sampling = np.zeros((runs, num_batch))
discounted_reward_off_policy_multiple_importance_sampling_cv = np.zeros((runs, num_batch))
discounted_reward_off_policy_multiple_importance_sampling_cv_baseline = np.zeros((runs, num_batch))
discounted_reward_off_policy_multiple_importance_sampling_pd = np.zeros((runs, num_batch))
discounted_reward_off_policy_multiple_importance_sampling_cv_pd = np.zeros((runs, num_batch))
discounted_reward_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated = np.zeros((runs, num_batch))
discounted_reward_off_policy_multiple_importance_sampling_cv_pd_baseline = np.zeros((runs, num_batch))
discounted_reward_reinforce = np.zeros((runs, num_batch))

policy_param_off_policy_importance_sampling = np.zeros((runs, num_batch))
policy_param_off_policy_importance_sampling_pd = np.zeros((runs, num_batch))
policy_param_off_policy_multiple_importance_sampling = np.zeros((runs, num_batch))
policy_param_off_policy_multiple_importance_sampling_cv = np.zeros((runs, num_batch))
policy_param_off_policy_multiple_importance_sampling_cv_baseline = np.zeros((runs, num_batch))
policy_param_off_policy_multiple_importance_sampling_pd = np.zeros((runs, num_batch))
policy_param_off_policy_multiple_importance_sampling_cv_pd = np.zeros((runs, num_batch))
policy_param_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated = np.zeros((runs, num_batch))
policy_param_off_policy_multiple_importance_sampling_cv_pd_baseline = np.zeros((runs, num_batch))
policy_param_reinforce = np.zeros((runs, num_batch))

gradient_off_policy_importance_sampling = np.zeros((runs, num_batch))
gradient_off_policy_importance_sampling_pd = np.zeros((runs, num_batch))
gradient_off_policy_multiple_importance_sampling = np.zeros((runs, num_batch))
gradient_off_policy_multiple_importance_sampling_cv = np.zeros((runs, num_batch))
gradient_off_policy_multiple_importance_sampling_cv_baseline = np.zeros((runs, num_batch))
gradient_off_policy_multiple_importance_sampling_pd = np.zeros((runs, num_batch))
gradient_off_policy_multiple_importance_sampling_cv_pd = np.zeros((runs, num_batch))
gradient_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated = np.zeros((runs, num_batch))
gradient_off_policy_multiple_importance_sampling_cv_pd_baseline = np.zeros((runs, num_batch))
gradient_reinforce = np.zeros((runs, num_batch))

ess_off_policy_importance_sampling = np.zeros((runs, num_batch))
ess_off_policy_importance_sampling_pd = np.zeros((runs, num_batch))
ess_off_policy_multiple_importance_sampling = np.zeros((runs, num_batch))
ess_off_policy_multiple_importance_sampling_cv = np.zeros((runs, num_batch))
ess_off_policy_multiple_importance_sampling_cv_baseline = np.zeros((runs, num_batch))
ess_off_policy_multiple_importance_sampling_pd = np.zeros((runs, num_batch))
ess_off_policy_multiple_importance_sampling_cv_pd = np.zeros((runs, num_batch))
ess_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated = np.zeros((runs, num_batch))
ess_off_policy_multiple_importance_sampling_cv_pd_baseline = np.zeros((runs, num_batch))

n_def_off_policy_importance_sampling = np.zeros((runs, num_batch))
n_def_off_policy_importance_sampling_pd = np.zeros((runs, num_batch))
n_def_off_policy_multiple_importance_sampling = np.zeros((runs, num_batch))
n_def_off_policy_multiple_importance_sampling_cv = np.zeros((runs, num_batch))
n_def_off_policy_multiple_importance_sampling_cv_baseline = np.zeros((runs, num_batch))
n_def_off_policy_multiple_importance_sampling_pd = np.zeros((runs, num_batch))
n_def_off_policy_multiple_importance_sampling_cv_pd = np.zeros((runs, num_batch))
n_def_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated = np.zeros((runs, num_batch))
n_def_off_policy_multiple_importance_sampling_cv_pd_baseline = np.zeros((runs, num_batch))

episodes_per_configuration = 20
env_param_min = 0.5
env_param_max = 1.5
policy_param_min = -1
policy_param_max = -0.1
linspace_policy = 10
linspace_env = 11
n_config_cv = (linspace_policy * linspace_env) - 1 #number of configurations to use to fit the control variates

seeds = [np.random.randint(1000000) for _ in range(runs)]

results = Parallel(n_jobs=10)(delayed(simulation)(env, batch_size, discount_factor, variance_action, episode_length, mean_initial_param, variance_initial_param, num_batch, learning_rate, seed, episodes_per_configuration, env_param_min, env_param_max, policy_param_min, policy_param_max) for seed in seeds)

for i_run in range(runs):

    discounted_reward_off_policy_importance_sampling[i_run, :] = results[i_run][0].disc_rewards
    discounted_reward_off_policy_importance_sampling_pd[i_run, :] = results[i_run][1].disc_rewards
    discounted_reward_reinforce[i_run, :] = results[i_run][2].disc_rewards
    discounted_reward_off_policy_multiple_importance_sampling[i_run, :] = results[i_run][3].disc_rewards
    discounted_reward_off_policy_multiple_importance_sampling_cv[i_run, :] = results[i_run][4].disc_rewards
    discounted_reward_off_policy_multiple_importance_sampling_cv_baseline[i_run, :] = results[i_run][5].disc_rewards
    discounted_reward_off_policy_multiple_importance_sampling_pd[i_run, :] = results[i_run][6].disc_rewards
    discounted_reward_off_policy_multiple_importance_sampling_cv_pd[i_run, :] = results[i_run][7].disc_rewards
    discounted_reward_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated[i_run, :] = results[i_run][8].disc_rewards
    discounted_reward_off_policy_multiple_importance_sampling_cv_pd_baseline[i_run, :] = results[i_run][9].disc_rewards

    policy_param_off_policy_importance_sampling[i_run, :] = results[i_run][0].policy_parameter
    policy_param_off_policy_importance_sampling_pd[i_run, :] = results[i_run][1].policy_parameter
    policy_param_reinforce[i_run, :] = results[i_run][2].policy_parameter
    policy_param_off_policy_multiple_importance_sampling[i_run, :] = results[i_run][3].policy_parameter
    policy_param_off_policy_multiple_importance_sampling_cv[i_run, :] = results[i_run][4].policy_parameter
    policy_param_off_policy_multiple_importance_sampling_cv_baseline[i_run, :] = results[i_run][5].policy_parameter
    policy_param_off_policy_multiple_importance_sampling_pd[i_run, :] = results[i_run][6].policy_parameter
    policy_param_off_policy_multiple_importance_sampling_cv_pd[i_run, :] = results[i_run][7].policy_parameter
    policy_param_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated[i_run, :] = results[i_run][8].policy_parameter
    policy_param_off_policy_multiple_importance_sampling_cv_pd_baseline[i_run, :] = results[i_run][9].policy_parameter

    gradient_off_policy_importance_sampling[i_run, :] = results[i_run][0].gradient
    gradient_off_policy_importance_sampling_pd[i_run, :] = results[i_run][1].gradient
    gradient_reinforce[i_run, :] = results[i_run][2].gradient
    gradient_off_policy_multiple_importance_sampling[i_run, :] = results[i_run][3].gradient
    gradient_off_policy_multiple_importance_sampling_cv[i_run, :] = results[i_run][4].gradient
    gradient_off_policy_multiple_importance_sampling_cv_baseline[i_run, :] = results[i_run][5].gradient
    gradient_off_policy_multiple_importance_sampling_pd[i_run, :] = results[i_run][6].gradient
    gradient_off_policy_multiple_importance_sampling_cv_pd[i_run, :] = results[i_run][7].gradient
    gradient_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated[i_run, :] = results[i_run][8].gradient
    gradient_off_policy_multiple_importance_sampling_cv_pd_baseline[i_run, :] = results[i_run][9].gradient

    ess_off_policy_importance_sampling[i_run, :] = results[i_run][0].ess
    ess_off_policy_importance_sampling_pd[i_run, :] = results[i_run][1].ess
    ess_off_policy_multiple_importance_sampling[i_run, :] = results[i_run][3].ess
    ess_off_policy_multiple_importance_sampling_cv[i_run, :] = results[i_run][4].ess
    ess_off_policy_multiple_importance_sampling_cv_baseline[i_run, :] = results[i_run][5].ess
    ess_off_policy_multiple_importance_sampling_pd[i_run, :] = results[i_run][6].ess
    ess_off_policy_multiple_importance_sampling_cv_pd[i_run, :] = results[i_run][7].ess
    ess_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated[i_run, :] = results[i_run][8].ess
    ess_off_policy_multiple_importance_sampling_cv_pd_baseline[i_run, :] = results[i_run][9].ess

    n_def_off_policy_importance_sampling[i_run, :] = results[i_run][0].n_def
    n_def_off_policy_importance_sampling_pd[i_run, :] = results[i_run][1].n_def
    n_def_off_policy_multiple_importance_sampling[i_run, :] = results[i_run][3].n_def
    n_def_off_policy_multiple_importance_sampling_cv[i_run, :] = results[i_run][4].n_def
    n_def_off_policy_multiple_importance_sampling_cv_baseline[i_run, :] = results[i_run][5].n_def
    n_def_off_policy_multiple_importance_sampling_pd[i_run, :] = results[i_run][6].n_def
    n_def_off_policy_multiple_importance_sampling_cv_pd[i_run, :] = results[i_run][7].n_def
    n_def_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated[i_run, :] = results[i_run][8].n_def
    n_def_off_policy_multiple_importance_sampling_cv_pd_baseline[i_run, :] = results[i_run][9].n_def

print("Saving files")
np.savetxt("discounted_reward_off_policy_importance_sampling.csv", discounted_reward_off_policy_importance_sampling, delimiter=",")
np.savetxt("discounted_reward_off_policy_importance_sampling_pd.csv", discounted_reward_off_policy_importance_sampling_pd, delimiter=",")
np.savetxt("discounted_reward_off_policy_multiple_importance_sampling.csv", discounted_reward_off_policy_multiple_importance_sampling, delimiter=",")
np.savetxt("discounted_reward_off_policy_multiple_importance_sampling_cv.csv", discounted_reward_off_policy_multiple_importance_sampling_cv, delimiter=",")
np.savetxt("discounted_reward_off_policy_multiple_importance_sampling_cv_baseline.csv", discounted_reward_off_policy_multiple_importance_sampling_cv_baseline, delimiter=",")
np.savetxt("discounted_reward_off_policy_multiple_importance_sampling_pd.csv", discounted_reward_off_policy_multiple_importance_sampling_pd, delimiter=",")
np.savetxt("discounted_reward_off_policy_multiple_importance_sampling_cv_pd.csv", discounted_reward_off_policy_multiple_importance_sampling_cv_pd, delimiter=",")
np.savetxt("discounted_reward_off_policy_multiple_importance_sampling_cv_pd_baseline.csv", discounted_reward_off_policy_multiple_importance_sampling_cv_pd_baseline, delimiter=",")
np.savetxt("discounted_reward_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated.csv", discounted_reward_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated, delimiter=",")
np.savetxt("discounted_reward_reinforce.csv", discounted_reward_reinforce, delimiter=",")

np.savetxt("policy_param_off_policy_importance_sampling.csv", policy_param_off_policy_importance_sampling, delimiter=",")
np.savetxt("policy_param_off_policy_importance_sampling_pd.csv", policy_param_off_policy_importance_sampling_pd, delimiter=",")
np.savetxt("policy_param_off_policy_multiple_importance_sampling.csv", policy_param_off_policy_multiple_importance_sampling, delimiter=",")
np.savetxt("policy_param_off_policy_multiple_importance_sampling_cv.csv", policy_param_off_policy_multiple_importance_sampling_cv, delimiter=",")
np.savetxt("policy_param_off_policy_multiple_importance_sampling_cv_baseline.csv", policy_param_off_policy_multiple_importance_sampling_cv_baseline, delimiter=",")
np.savetxt("policy_param_off_policy_multiple_importance_sampling_pd.csv", policy_param_off_policy_multiple_importance_sampling_pd, delimiter=",")
np.savetxt("policy_param_off_policy_multiple_importance_sampling_cv_pd.csv", policy_param_off_policy_multiple_importance_sampling_cv_pd, delimiter=",")
np.savetxt("policy_param_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated.csv", policy_param_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated, delimiter=",")
np.savetxt("policy_param_off_policy_multiple_importance_sampling_cv_pd_baseline.csv", policy_param_off_policy_multiple_importance_sampling_cv_pd_baseline, delimiter=",")
np.savetxt("policy_param_reinforce.csv", policy_param_reinforce, delimiter=",")

np.savetxt("gradient_off_policy_importance_sampling.csv", gradient_off_policy_importance_sampling, delimiter=",")
np.savetxt("gradient_off_policy_importance_sampling_pd.csv", gradient_off_policy_importance_sampling_pd, delimiter=",")
np.savetxt("gradient_off_policy_multiple_importance_sampling.csv", gradient_off_policy_multiple_importance_sampling, delimiter=",")
np.savetxt("gradient_off_policy_multiple_importance_sampling_cv.csv", gradient_off_policy_multiple_importance_sampling_cv, delimiter=",")
np.savetxt("gradient_off_policy_multiple_importance_sampling_cv_baseline.csv", gradient_off_policy_multiple_importance_sampling_cv_baseline, delimiter=",")
np.savetxt("gradient_off_policy_multiple_importance_sampling_pd.csv", gradient_off_policy_multiple_importance_sampling_pd, delimiter=",")
np.savetxt("gradient_off_policy_multiple_importance_sampling_cv_pd.csv", gradient_off_policy_multiple_importance_sampling_cv_pd, delimiter=",")
np.savetxt("gradient_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated.csv", gradient_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated, delimiter=",")
np.savetxt("gradient_off_policy_multiple_importance_sampling_cv_pd_baseline.csv", gradient_off_policy_multiple_importance_sampling_cv_pd_baseline, delimiter=",")
np.savetxt("gradient_reinforce.csv", gradient_reinforce, delimiter=",")

np.savetxt("ess_off_policy_importance_sampling.csv", ess_off_policy_importance_sampling, delimiter=",")
np.savetxt("ess_off_policy_importance_sampling_pd.csv", ess_off_policy_importance_sampling_pd, delimiter=",")
np.savetxt("ess_off_policy_multiple_importance_sampling.csv", ess_off_policy_multiple_importance_sampling, delimiter=",")
np.savetxt("ess_off_policy_multiple_importance_sampling_cv.csv", ess_off_policy_multiple_importance_sampling_cv, delimiter=",")
np.savetxt("ess_off_policy_multiple_importance_sampling_cv_baseline.csv", ess_off_policy_multiple_importance_sampling_cv_baseline, delimiter=",")
np.savetxt("ess_off_policy_multiple_importance_sampling_pd.csv", ess_off_policy_multiple_importance_sampling_pd, delimiter=",")
np.savetxt("ess_off_policy_multiple_importance_sampling_cv_pd.csv", ess_off_policy_multiple_importance_sampling_cv_pd, delimiter=",")
np.savetxt("ess_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated.csv", ess_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated, delimiter=",")
np.savetxt("ess_off_policy_multiple_importance_sampling_cv_pd_baseline.csv", ess_off_policy_multiple_importance_sampling_cv_pd_baseline, delimiter=",")

np.savetxt("n_def_off_policy_importance_sampling.csv", n_def_off_policy_importance_sampling, delimiter=",")
np.savetxt("n_def_off_policy_importance_sampling_pd.csv", n_def_off_policy_importance_sampling_pd, delimiter=",")
np.savetxt("n_def_off_policy_multiple_importance_sampling.csv", n_def_off_policy_multiple_importance_sampling, delimiter=",")
np.savetxt("n_def_off_policy_multiple_importance_sampling_cv.csv", n_def_off_policy_multiple_importance_sampling_cv, delimiter=",")
np.savetxt("n_def_off_policy_multiple_importance_sampling_cv_baseline.csv", n_def_off_policy_multiple_importance_sampling_cv_baseline, delimiter=",")
np.savetxt("n_def_off_policy_multiple_importance_sampling_pd.csv", n_def_off_policy_multiple_importance_sampling_pd, delimiter=",")
np.savetxt("n_def_off_policy_multiple_importance_sampling_cv_pd.csv", n_def_off_policy_multiple_importance_sampling_cv_pd, delimiter=",")
np.savetxt("n_def_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated.csv", n_def_off_policy_multiple_importance_sampling_cv_pd_baseline_approximated, delimiter=",")
np.savetxt("n_def_off_policy_multiple_importance_sampling_cv_pd_baseline.csv", n_def_off_policy_multiple_importance_sampling_cv_pd_baseline, delimiter=",")

stats_opt = iw.optimalPolicy(env, num_batch, 10, discount_factor, variance_action, episode_length) # Optimal policy

np.savetxt("discounted_reward_optimal.csv", stats_opt.disc_rewards, delimiter=",")
np.savetxt("policy_param_optimal.csv", stats_opt.policy_parameter, delimiter=",")
