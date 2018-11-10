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
num_episodes = 1500
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

# print("Learn policy")
# for i_run in range(runs):
#
#     print(i_run)
#     initial_param = np.random.normal(mean_initial_param, variance_initial_param)
#
#     off_policy_importance_sampling = iw.offPolicyImportanceSampling(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
#     off_policy_importance_sampling_pd = iw.offPolicyImportanceSamplingPd(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
#     reinforce = alg.reinforce(env, num_episodes, batch_size, discount_factor, episode_length, initial_param, variance_action)
#     off_policy_multiple_importance_sampling = iw.offPolicyMultipleImportanceSampling(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
#     #off_policy_multiple_importance_sampling_pd = iw.offPolicyMultipleImportanceSamplingPd(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
#
#     discounted_reward_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.episode_disc_rewards
#     discounted_reward_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.episode_disc_rewards
#     discounted_reward_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.episode_disc_rewards
#     #discounted_reward_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.episode_disc_rewards
#     discounted_reward_reinforce[i_run, :] = reinforce.episode_disc_rewards
#
#     policy_param_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.policy_parameter
#     policy_param_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.policy_parameter
#     policy_param_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.policy_parameter
#     #policy_param_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.policy_parameter
#     policy_param_reinforce[i_run, :] = reinforce.policy_parameter
#
#     gradient_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.gradient
#     gradient_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.gradient
#     gradient_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.gradient
#     #gradient_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.gradient
#     gradient_reinforce[i_run, :] = reinforce.gradient
#
#     ess_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.ess
#     ess_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.ess
#     ess_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.ess
#     #ess_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.ess

discounted_reward_off_policy_importance_sampling = np.genfromtxt('discounted_reward_off_policy_importance_sampling.csv', delimiter=',')
discounted_reward_off_policy_importance_sampling_pd = np.genfromtxt('discounted_reward_off_policy_importance_sampling_pd.csv', delimiter=',')
discounted_reward_off_policy_multiple_importance_sampling = np.genfromtxt('discounted_reward_off_policy_multiple_importance_sampling.csv', delimiter=',')
discounted_reward_reinforce = np.genfromtxt('discounted_reward_reinforce.csv', delimiter=',')

policy_param_off_policy_importance_sampling = np.genfromtxt('policy_param_off_policy_importance_sampling.csv', delimiter=',')
policy_param_off_policy_importance_sampling_pd = np.genfromtxt('policy_param_off_policy_importance_sampling_pd.csv', delimiter=',')
policy_param_off_policy_multiple_importance_sampling = np.genfromtxt('policy_param_off_policy_multiple_importance_sampling.csv', delimiter=',')
policy_param_reinforce = np.genfromtxt('policy_param_reinforce.csv', delimiter=',')

gradient_off_policy_importance_sampling = np.genfromtxt('gradient_off_policy_importance_sampling.csv', delimiter=',')
gradient_off_policy_importance_sampling_pd = np.genfromtxt('gradient_off_policy_importance_sampling_pd.csv', delimiter=',')
gradient_off_policy_multiple_importance_sampling = np.genfromtxt('gradient_off_policy_multiple_importance_sampling.csv', delimiter=',')
gradient_reinforce = np.genfromtxt('gradient_reinforce.csv', delimiter=',')

ess_off_policy_importance_sampling = np.genfromtxt('ess_off_policy_importance_sampling.csv', delimiter=',')
ess_off_policy_importance_sampling_pd = np.genfromtxt('ess_off_policy_importance_sampling_pd.csv', delimiter=',')
#ess_off_policy_multiple_importance_sampling = np.genfromtxt('ess_off_policy_multiple_importance_sampling.csv', delimiter=',')

# print("Saving files")
# np.savetxt("discounted_reward_off_policy_importance_sampling.csv", discounted_reward_off_policy_importance_sampling, delimiter=",")
# np.savetxt("discounted_reward_off_policy_importance_sampling_pd.csv", discounted_reward_off_policy_importance_sampling_pd, delimiter=",")
# np.savetxt("discounted_reward_off_policy_multiple_importance_sampling.csv", discounted_reward_off_policy_multiple_importance_sampling, delimiter=",")
# #np.savetxt("discounted_reward_off_policy_multiple_importance_sampling_pd.csv", discounted_reward_off_policy_multiple_importance_sampling_pd, delimiter=",")
# np.savetxt("discounted_reward_reinforce.csv", discounted_reward_reinforce, delimiter=",")
#
# np.savetxt("policy_param_off_policy_importance_sampling.csv", policy_param_off_policy_importance_sampling, delimiter=",")
# np.savetxt("policy_param_off_policy_importance_sampling_pd.csv", policy_param_off_policy_importance_sampling_pd, delimiter=",")
# np.savetxt("policy_param_off_policy_multiple_importance_sampling.csv", policy_param_off_policy_multiple_importance_sampling, delimiter=",")
# #np.savetxt("policy_param_off_policy_multiple_importance_sampling_pd.csv", policy_param_off_policy_multiple_importance_sampling_pd, delimiter=",")
# np.savetxt("policy_param_reinforce.csv", policy_param_reinforce, delimiter=",")
#
# np.savetxt("gradient_off_policy_importance_sampling.csv", gradient_off_policy_importance_sampling, delimiter=",")
# np.savetxt("gradient_off_policy_importance_sampling_pd.csv", gradient_off_policy_importance_sampling_pd, delimiter=",")
# np.savetxt("gradient_off_policy_multiple_importance_sampling.csv", gradient_off_policy_multiple_importance_sampling, delimiter=",")
# #np.savetxt("gradient_off_policy_multiple_importance_sampling_pd.csv", gradient_off_policy_multiple_importance_sampling_pd, delimiter=",")
# np.savetxt("gradient_reinforce.csv", gradient_reinforce, delimiter=",")
#
# np.savetxt("ess_off_policy_importance_sampling.csv", ess_off_policy_importance_sampling, delimiter=",")
# np.savetxt("ess_off_policy_importance_sampling_pd.csv", ess_off_policy_importance_sampling_pd, delimiter=",")
# np.savetxt("ess_off_policy_multiple_importance_sampling.csv", ess_off_policy_multiple_importance_sampling, delimiter=",")
# #np.savetxt("ess_off_policy_multiple_importance_sampling_pd.csv", ess_off_policy_multiple_importance_sampling_pd, delimiter=",")

stats_opt = iw.optimalPolicy(env, num_episodes, discount_factor, batch_size, episode_length) # Optimal policy

#Preparing plots
x = range(num_batch)

# Discounted rewards
mean_alg1 = np.mean(discounted_reward_off_policy_importance_sampling, axis=0)
mean_alg2 = np.mean(discounted_reward_off_policy_importance_sampling_pd, axis=0)
mean_alg3 = np.mean(discounted_reward_off_policy_multiple_importance_sampling, axis=0)
mean_alg4 = np.mean(discounted_reward_off_policy_multiple_importance_sampling_pd, axis=0)
mean_alg5 = np.mean(discounted_reward_reinforce, axis=0)

var_alg1 = np.std(discounted_reward_off_policy_importance_sampling, axis=0) / (m.sqrt(runs))
var_alg2 = np.std(discounted_reward_off_policy_importance_sampling_pd, axis=0) / (m.sqrt(runs))
var_alg3 = np.std(discounted_reward_off_policy_multiple_importance_sampling, axis=0) / (m.sqrt(runs))
var_alg4 = np.std(discounted_reward_off_policy_multiple_importance_sampling_pd, axis=0) / (m.sqrt(runs))
var_alg5 = np.std(discounted_reward_reinforce, axis=0) / (m.sqrt(runs))

plot.plot_curves([x, x, x, x, x], [mean_alg1, mean_alg2, mean_alg3, mean_alg4, mean_alg5], [var_alg1, var_alg2, var_alg3, var_alg4, var_alg5], title = "Rewards over batches", x_label = "Batch", y_label = "Discounted reward", names = ["REINFORCE with transfer (IS)", "REINFORCE with transfer (PD-IS)", "REINFORCE with transfer (MIS)", "REINFORCE with transfer (PD-MIS)", "REINFORCE"], file_name = "Rewards transfer policy search")

# Policy parameters
mean_alg1 = np.mean(policy_param_off_policy_importance_sampling, axis=0)
mean_alg2 = np.mean(policy_param_off_policy_importance_sampling_pd, axis=0)
mean_alg3 = np.mean(policy_param_off_policy_multiple_importance_sampling, axis=0)
mean_alg4 = np.mean(policy_param_off_policy_multiple_importance_sampling_pd, axis=0)
mean_alg5 = np.mean(policy_param_reinforce, axis=0)

var_alg1 = np.std(policy_param_off_policy_importance_sampling, axis=0) / (m.sqrt(runs))
var_alg2 = np.std(policy_param_off_policy_importance_sampling_pd, axis=0) / (m.sqrt(runs))
var_alg3 = np.std(policy_param_off_policy_multiple_importance_sampling, axis=0) / (m.sqrt(runs))
var_alg4 = np.std(policy_param_off_policy_multiple_importance_sampling_pd, axis=0) / (m.sqrt(runs))
var_alg5 = np.std(policy_param_reinforce, axis=0) / (m.sqrt(runs))

plot.plot_curves([x, x, x, x, x], [mean_alg1, mean_alg2, mean_alg3, mean_alg4, mean_alg5], [var_alg1, var_alg2, var_alg3, var_alg4, var_alg5], title = "Policy param over batches", x_label = "Batch", y_label = "Policy param", names = ["REINFORCE with transfer (IS)", "REINFORCE with transfer (PD-IS)", "REINFORCE with transfer (MIS)", "REINFORCE with transfer (PD-MIS)", "REINFORCE"], file_name = "Policy param transfer policy search")

# Gradients
mean_pol1 = np.mean(gradient_off_policy_importance_sampling, axis=0)
mean_pol2 = np.mean(gradient_off_policy_importance_sampling_pd, axis=0)
mean_pol3 = np.mean(gradient_off_policy_multiple_importance_sampling, axis=0)
mean_pol4 = np.mean(gradient_off_policy_multiple_importance_sampling_pd, axis=0)
mean_pol5 = np.mean(gradient_reinforce, axis=0)

var_pol1 = np.std(gradient_off_policy_importance_sampling, axis=0) / (m.sqrt(runs))
var_pol2 = np.std(gradient_off_policy_importance_sampling_pd, axis=0) / (m.sqrt(runs))
var_pol3 = np.std(gradient_off_policy_multiple_importance_sampling, axis=0) / (m.sqrt(runs))
var_pol4 = np.std(gradient_off_policy_multiple_importance_sampling_pd, axis=0) / (m.sqrt(runs))
var_pol5 = np.std(gradient_reinforce, axis=0) / (m.sqrt(runs))

plot.plot_curves([x, x, x, x, x], [mean_alg1, mean_alg2, mean_alg3, mean_alg4, mean_alg5], [var_alg1, var_alg2, var_alg3, var_alg4, var_alg5], title = "Gradient over batches", x_label = "Batch", y_label = "Gradient", names = ["REINFORCE with transfer (IS)", "REINFORCE with transfer (PD-IS)", "REINFORCE with transfer (MIS)", "REINFORCE with transfer (PD-MIS)", "REINFORCE"], file_name = "Gradients transfer policy search")

# Effective Sample Size
mean_pol1 = np.mean(ess_off_policy_importance_sampling, axis=0)
mean_pol2 = np.mean(ess_off_policy_importance_sampling_pd, axis=0)
mean_pol3 = np.mean(ess_off_policy_multiple_importance_sampling, axis=0)
mean_pol4 = np.mean(ess_off_policy_multiple_importance_sampling_pd, axis=0)

var_pol1 = np.std(ess_off_policy_importance_sampling, axis=0) / (m.sqrt(runs))
var_pol2 = np.std(ess_off_policy_importance_sampling_pd, axis=0) / (m.sqrt(runs))
var_pol3 = np.std(ess_off_policy_multiple_importance_sampling, axis=0) / (m.sqrt(runs))
var_pol4 = np.std(ess_off_policy_multiple_importance_sampling_pd, axis=0) / (m.sqrt(runs))

plot.plot_curves([x, x, x, x], [mean_alg1, mean_alg2, mean_alg3, mean_alg4], [var_alg1, var_alg2, var_alg3, var_alg4], title = "ESS over batches", x_label = "Batch", y_label = "ESS", names = ["REINFORCE with transfer (IS)", "REINFORCE with transfer (PD-IS)", "REINFORCE with transfer (MIS)", "REINFORCE with transfer (PD-MIS)"], file_name = "ESS transfer policy search")
