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
num_episodes = 500
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

print("Learn policy")
for i_run in range(runs):

    print(i_run)
    initial_param = np.random.normal(mean_initial_param, variance_initial_param)

    #off_policy_importance_sampling = iw.offPolicyImportanceSampling(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
    #off_policy_importance_sampling_pd = iw.offPolicyImportanceSamplingPd(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
    #reinforce = alg.reinforce(env, num_episodes, batch_size, discount_factor, episode_length, initial_param, variance_action)
    off_policy_multiple_importance_sampling = iw.offPolicyMultipleImportanceSampling(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
    #off_policy_multiple_importance_sampling_pd = iw.offPolicyMultipleImportanceSamplingPd(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)

    #discounted_reward_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.episode_disc_rewards
    #discounted_reward_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.episode_disc_rewards
    discounted_reward_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.episode_disc_rewards
    #discounted_reward_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.episode_disc_rewards
    #discounted_reward_reinforce[i_run, :] = reinforce.episode_disc_rewards

    #policy_param_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.policy_parameter
    #policy_param_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.policy_parameter
    policy_param_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.policy_parameter
    #policy_param_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.policy_parameter
    #policy_param_reinforce[i_run, :] = reinforce.policy_parameter

# discounted_reward_off_policy = np.genfromtxt('discounted_reward_off_policy_importance_sampling.csv', delimiter=',')
# discounted_reward_off_policy = np.genfromtxt('discounted_reward_off_policy_importance_sampling_pd.csv', delimiter=',')
# discounted_reward_reinforce = np.genfromtxt('discounted_reward_reinforce.csv', delimiter=',')
# policy_param_off_policy = np.genfromtxt('policy_param_off_policy_importance_sampling.csv', delimiter=',')
# policy_param_off_policy = np.genfromtxt('policy_param_off_policy_importance_sampling_pd.csv', delimiter=',')
# policy_param_reinforce = np.genfromtxt('policy_param_reinforce.csv', delimiter=',')

stats_opt = iw.optimalPolicy(env, num_episodes, discount_factor, batch_size, episode_length) # Optimal policy

print("Saving files")
#np.savetxt("discounted_reward_off_policy_importance_sampling.csv", discounted_reward_off_policy_importance_sampling, delimiter=",")
#np.savetxt("discounted_reward_off_policy_importance_sampling_pd.csv", discounted_reward_off_policy_importance_sampling_pd, delimiter=",")
np.savetxt("discounted_reward_off_policy_multiple_importance_sampling.csv", discounted_reward_off_policy_multiple_importance_sampling, delimiter=",")
#np.savetxt("discounted_reward_off_policy_multiple_importance_sampling_pd.csv", discounted_reward_off_policy_multiple_importance_sampling_pd, delimiter=",")
#np.savetxt("discounted_reward_reinforce.csv", discounted_reward_reinforce, delimiter=",")
#np.savetxt("policy_param_off_policy_importance_sampling.csv", policy_param_off_policy_importance_sampling, delimiter=",")
np.savetxt("policy_param_off_policy_importance_sampling_pd.csv", policy_param_off_policy_importance_sampling_pd, delimiter=",")
#np.savetxt("policy_param_off_policy_multiple_importance_sampling.csv", policy_param_off_policy_multiple_importance_sampling, delimiter=",")
#np.savetxt("policy_param_off_policy_multiple_importance_sampling_pd.csv", policy_param_off_policy_multiple_importance_sampling_pd, delimiter=",")
np.savetxt("policy_param_reinforce.csv", policy_param_reinforce, delimiter=",")

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


mean_pol1 = np.mean(policy_param_off_policy_importance_sampling, axis=0)
mean_pol2 = np.mean(policy_param_off_policy_importance_sampling_pd, axis=0)
mean_pol3 = np.mean(policy_param_off_policy_multiple_importance_sampling, axis=0)
mean_pol4 = np.mean(policy_param_off_policy_multiple_importance_sampling_pd, axis=0)
mean_pol5 = np.mean(policy_param_reinforce, axis=0)

var_pol1 = np.std(policy_param_off_policy_importance_sampling, axis=0) / (m.sqrt(runs))
var_pol2 = np.std(policy_param_off_policy_importance_sampling_pd, axis=0) / (m.sqrt(runs))
var_pol3 = np.std(policy_param_off_policy_multiple_importance_sampling, axis=0) / (m.sqrt(runs))
var_pol4 = np.std(policy_param_off_policy_multiple_importance_sampling_pd, axis=0) / (m.sqrt(runs))
var_pol5 = np.std(policy_param_reinforce, axis=0) / (m.sqrt(runs))

x = range(num_batch)

plot.plot_curves([x, x, x, x, x] , [mean_alg1, mean_alg2, mean_alg3, mean_alg4, mean_alg5], [var_alg1, var_alg2, var_alg3, var_alg4, var_alg5], title = "Rewards over batches", x_label = "Batch", y_label = "Discounted reward", names = ["REINFORCE with transfer (IS)", "REINFORCE with transfer (PD-IS)", "REINFORCE with transfer (MIS)", "REINFORCE with transfer (PD-MIS)", "REINFORCE"], file_name = "Rewards transfer policy search")
plot.plot_curves([x, x, x, x, x] , [mean_pol1, mean_pol2, mean_pol3, mean_pol4, mean_pol5], [var_pol1, var_pol2, var_alg3, var_pol4, var_pol5], title = "Policy parameter over batches", x_label = "Batch", y_label = "Policy parameter", names = ["REINFORCE with transfer (IS)", "REINFORCE with transfer (PD-IS)", "REINFORCE with transfer (MIS)", "REINFORCE with transfer (PD-MIS)", "REINFORCE"], file_name = "Policy parameter transfer policy search")
