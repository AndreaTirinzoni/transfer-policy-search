import gym
import envs
import numpy as np
import algorithmPolicySearch as alg
import importanceWeights as iw
import math as m
from utils import plot

np.set_printoptions(precision=4)
env = gym.make('LQG1D-v0')

mean_initial_param = -0.1
episode_length = 20
variance_initial_param = 0
variance_action = 0.1
num_episodes = 1200
batch_size = 10
num_batch = num_episodes//batch_size
discount_factor = 0.99
runs = 5

print("Loading files")
source_task = np.genfromtxt('source_task.csv', delimiter=',')
episodes_per_config = np.genfromtxt('episodes_per_config.csv', delimiter=',').astype(int)
source_param = np.genfromtxt('source_param.csv', delimiter=',')

discounted_reward_off_policy_importance_sampling = np.zeros((runs, num_batch))
discounted_reward_off_policy_importance_sampling_pd = np.zeros((runs, num_batch))
discounted_reward_off_policy_multiple_importance_sampling = np.zeros((runs, num_batch))
discounted_reward_off_policy_multiple_importance_sampling_pd = np.zeros((runs, num_batch))
discounted_reward_reinforce = np.zeros((runs, num_batch))

policy_param_off_policy_importance_sampling = np.zeros((runs, num_batch))
policy_param_off_policy_importance_sampling_pd = np.zeros((runs, num_batch))
policy_param_off_policy_multiple_importance_sampling = np.zeros((runs, num_batch))
policy_param_off_policy_multiple_importance_sampling_pd = np.zeros((runs, num_batch))
policy_param_reinforce = np.zeros((runs, num_batch))

gradient_off_policy_importance_sampling = np.zeros((runs, num_batch))
gradient_off_policy_importance_sampling_pd = np.zeros((runs, num_batch))
gradient_off_policy_multiple_importance_sampling = np.zeros((runs, num_batch))
gradient_off_policy_multiple_importance_sampling_pd = np.zeros((runs, num_batch))
gradient_reinforce = np.zeros((runs, num_batch))

ess_off_policy_importance_sampling = np.zeros((runs, num_batch))
ess_off_policy_importance_sampling_pd = np.zeros((runs, num_batch))
ess_off_policy_multiple_importance_sampling = np.zeros((runs, num_batch))
ess_off_policy_multiple_importance_sampling_pd = np.zeros((runs, num_batch))

print("Learn policy")
for i_run in range(runs):

    print("Run: " + str(i_run))
    initial_param = np.random.normal(mean_initial_param, variance_initial_param)

    #off_policy_importance_sampling = iw.offPolicyImportanceSampling(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
    #off_policy_importance_sampling_pd = iw.offPolicyImportanceSamplingPd(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
    #reinforce = alg.reinforce(env, num_episodes, batch_size, discount_factor, episode_length, initial_param, variance_action)
    #off_policy_multiple_importance_sampling = iw.offPolicyMultipleImportanceSampling(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
    off_policy_multiple_importance_sampling_pd = iw.offPolicyMultipleImportanceSamplingPd(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)

    discounted_reward_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.episode_disc_rewards
    discounted_reward_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.episode_disc_rewards
    discounted_reward_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.episode_disc_rewards
    discounted_reward_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.episode_disc_rewards
    discounted_reward_reinforce[i_run, :] = reinforce.episode_disc_rewards

    policy_param_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.policy_parameter
    policy_param_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.policy_parameter
    policy_param_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.policy_parameter
    policy_param_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.policy_parameter
    policy_param_reinforce[i_run, :] = reinforce.policy_parameter

    gradient_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.gradient
    gradient_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.gradient
    gradient_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.gradient
    gradient_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.gradient
    gradient_reinforce[i_run, :] = reinforce.gradient

    ess_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.ess
    ess_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.ess
    ess_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.ess
    ess_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.ess

print("Saving files")
np.savetxt("discounted_reward_off_policy_importance_sampling.csv", discounted_reward_off_policy_importance_sampling, delimiter=",")
np.savetxt("discounted_reward_off_policy_importance_sampling_pd.csv", discounted_reward_off_policy_importance_sampling_pd, delimiter=",")
np.savetxt("discounted_reward_off_policy_multiple_importance_sampling.csv", discounted_reward_off_policy_multiple_importance_sampling, delimiter=",")
np.savetxt("discounted_reward_off_policy_multiple_importance_sampling_pd.csv", discounted_reward_off_policy_multiple_importance_sampling_pd, delimiter=",")
np.savetxt("discounted_reward_reinforce.csv", discounted_reward_reinforce, delimiter=",")

np.savetxt("policy_param_off_policy_importance_sampling.csv", policy_param_off_policy_importance_sampling, delimiter=",")
np.savetxt("policy_param_off_policy_importance_sampling_pd.csv", policy_param_off_policy_importance_sampling_pd, delimiter=",")
np.savetxt("policy_param_off_policy_multiple_importance_sampling.csv", policy_param_off_policy_multiple_importance_sampling, delimiter=",")
np.savetxt("policy_param_off_policy_multiple_importance_sampling_pd.csv", policy_param_off_policy_multiple_importance_sampling_pd, delimiter=",")
np.savetxt("policy_param_reinforce.csv", policy_param_reinforce, delimiter=",")

np.savetxt("gradient_off_policy_importance_sampling.csv", gradient_off_policy_importance_sampling, delimiter=",")
np.savetxt("gradient_off_policy_importance_sampling_pd.csv", gradient_off_policy_importance_sampling_pd, delimiter=",")
np.savetxt("gradient_off_policy_multiple_importance_sampling.csv", gradient_off_policy_multiple_importance_sampling, delimiter=",")#np.savetxt("gradient_off_policy_multiple_importance_sampling_pd.csv", gradient_off_policy_multiple_importance_sampling_pd, delimiter=",")
np.savetxt("gradient_off_policy_multiple_importance_sampling_pd.csv", gradient_off_policy_multiple_importance_sampling_pd, delimiter=",")
np.savetxt("gradient_reinforce.csv", gradient_reinforce, delimiter=",")

np.savetxt("ess_off_policy_importance_sampling.csv", ess_off_policy_importance_sampling, delimiter=",")
np.savetxt("ess_off_policy_importance_sampling_pd.csv", ess_off_policy_importance_sampling_pd, delimiter=",")
np.savetxt("ess_off_policy_multiple_importance_sampling.csv", ess_off_policy_multiple_importance_sampling, delimiter=",")
np.savetxt("ess_off_policy_multiple_importance_sampling_pd.csv", ess_off_policy_multiple_importance_sampling_pd, delimiter=",")

stats_opt = iw.optimalPolicy(env, num_episodes, batch_size, discount_factor, variance_action, episode_length) # Optimal policy

np.savetxt("discounted_reward_optimal.csv", stats_opt.episode_disc_rewards, delimiter=",")
np.savetxt("policy_param_optimal.csv", stats_opt.policy_parameter, delimiter=",")
