import gym
import envs
import numpy as np
import algorithmPolicySearch as alg
import importanceWeights as iw
from utils import plot
import math as m
import sourceTaskCreation as stc

np.set_printoptions(precision=4)
env = gym.make('LQG1D-v0')

mean_initial_param = -0.1
episode_length = 20
variance_initial_param = 0
variance_action = 0.1
batch_size = 10
num_batch = 200
discount_factor = 0.99
runs = 10
learning_rate = 0.1

episodes_per_configuration = 10
env_param_min = 0.5
env_param_max = 1.5
policy_param_min = -1
policy_param_max = 0

print("Loading files")
source_task = np.genfromtxt('source_task.csv', delimiter=',')
episodes_per_config = np.genfromtxt('episodes_per_config.csv', delimiter=',').astype(int)
source_param = np.genfromtxt('source_param.csv', delimiter=',')
next_states_unclipped = np.genfromtxt('next_states_unclipped.csv', delimiter=',')
actions_clipped = np.genfromtxt('actions_clipped.csv', delimiter=',')
#[source_task, source_param, episodes_per_config, next_states_unclipped, actions_clipped] = stc.sourceTaskCreation(episode_length, episodes_per_configuration, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max)

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

    #[source_task, source_param, episodes_per_config, next_states_unclipped, actions_clipped] = stc.sourceTaskCreation(episode_length, episodes_per_configuration, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max)

    #off_policy_importance_sampling = iw.offPolicyImportanceSampling(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
    #off_policy_importance_sampling_pd = iw.offPolicyImportanceSamplingPd(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
    #reinforce = alg.reinforce(env, num_batch, batch_size, discount_factor, episode_length, initial_param, variance_action)
    #off_policy_multiple_importance_sampling = iw.offPolicyMultipleImportanceSampling(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
    # print("MIS-CV")
    # off_policy_multiple_importance_sampling_cv = iw.offPolicyMultipleImportanceSamplingCv(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)

    # print("MIS-CV-BASELINE")
    # off_policy_multiple_importance_sampling_cv_baseline = iw.offPolicyMultipleImportanceSamplingCvBaseline(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)

    # print("PD-MIS-CV")
    # off_policy_multiple_importance_sampling_pd_cv = iw.offPolicyMultipleImportanceSamplingCvPd(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)

    print("PD-MIS-CV-BASELINE")
    off_policy_multiple_importance_sampling_pd_cv_baseline = iw.offPolicyMultipleImportanceSamplingCvPdBaselineApproximated(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, learning_rate)

    # discounted_reward_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.episode_disc_rewards
    # discounted_reward_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.episode_disc_rewards
    # discounted_reward_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.episode_disc_rewards
    # discounted_reward_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.episode_disc_rewards
    # discounted_reward_reinforce[i_run, :] = reinforce.episode_disc_rewards
    off_policy_multiple_importance_sampling_pd_cv_baseline = iw.offPolicyMultipleImportanceSamplingCvPdBaseline(env, batch_size, discount_factor, source_task, next_states_unclipped, actions_clipped, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, 0.000001)
    policy_param_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.policy_parameter
    policy_param_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.policy_parameter
    # policy_param_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.policy_parameter
    # policy_param_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.policy_parameter
    # policy_param_reinforce[i_run, :] = reinforce.policy_parameter
    #
    # gradient_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.gradient
    # gradient_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.gradient
    # gradient_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.gradient
    # gradient_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.gradient
    # gradient_reinforce[i_run, :] = reinforce.gradient
    #
    # ess_off_policy_importance_sampling[i_run,:] = off_policy_importance_sampling.ess
    # ess_off_policy_importance_sampling_pd[i_run,:] = off_policy_importance_sampling_pd.ess
    # ess_off_policy_multiple_importance_sampling[i_run,:] = off_policy_multiple_importance_sampling.ess
    # ess_off_policy_multiple_importance_sampling_pd[i_run,:] = off_policy_multiple_importance_sampling_pd.ess

# print("Saving files")
# np.savetxt("./20run/discounted_reward_off_policy_importance_sampling.csv", discounted_reward_off_policy_importance_sampling, delimiter=",")
# np.savetxt("./20run/discounted_reward_off_policy_importance_sampling_pd.csv", discounted_reward_off_policy_importance_sampling_pd, delimiter=",")
# np.savetxt("./20run/discounted_reward_off_policy_multiple_importance_sampling.csv", discounted_reward_off_policy_multiple_importance_sampling, delimiter=",")
# np.savetxt("./20run/discounted_reward_off_policy_multiple_importance_sampling_pd.csv", discounted_reward_off_policy_multiple_importance_sampling_pd, delimiter=",")
# np.savetxt("./20run/discounted_reward_reinforce.csv", discounted_reward_reinforce, delimiter=",")
#
# np.savetxt("./20run/policy_param_off_policy_importance_sampling.csv", policy_param_off_policy_importance_sampling, delimiter=",")
# np.savetxt("./20run/policy_param_off_policy_importance_sampling_pd.csv", policy_param_off_policy_importance_sampling_pd, delimiter=",")
# np.savetxt("./20run/policy_param_off_policy_multiple_importance_sampling.csv", policy_param_off_policy_multiple_importance_sampling, delimiter=",")
# np.savetxt("./20run/policy_param_off_policy_multiple_importance_sampling_pd.csv", policy_param_off_policy_multiple_importance_sampling_pd, delimiter=",")
# np.savetxt("./20run/policy_param_reinforce.csv", policy_param_reinforce, delimiter=",")
#
# np.savetxt("./20run/gradient_off_policy_importance_sampling.csv", gradient_off_policy_importance_sampling, delimiter=",")
# np.savetxt("./20run/gradient_off_policy_importance_sampling_pd.csv", gradient_off_policy_importance_sampling_pd, delimiter=",")
# np.savetxt("./20run/gradient_off_policy_multiple_importance_sampling.csv", gradient_off_policy_multiple_importance_sampling, delimiter=",")#np.savetxt("gradient_off_policy_multiple_importance_sampling_pd.csv", gradient_off_policy_multiple_importance_sampling_pd, delimiter=",")
# np.savetxt("./20run/gradient_off_policy_multiple_importance_sampling_pd.csv", gradient_off_policy_multiple_importance_sampling_pd, delimiter=",")
# np.savetxt("./20run/gradient_reinforce.csv", gradient_reinforce, delimiter=",")
#
# np.savetxt("./20run/ess_off_policy_importance_sampling.csv", ess_off_policy_importance_sampling, delimiter=",")
# np.savetxt("./20run/ess_off_policy_importance_sampling_pd.csv", ess_off_policy_importance_sampling_pd, delimiter=",")
# np.savetxt("./20run/ess_off_policy_multiple_importance_sampling.csv", ess_off_policy_multiple_importance_sampling, delimiter=",")
# np.savetxt("./20run/ess_off_policy_multiple_importance_sampling_pd.csv", ess_off_policy_multiple_importance_sampling_pd, delimiter=",")
#
# stats_opt = iw.optimalPolicy(env, num_batch, batch_size, discount_factor, variance_action, episode_length) # Optimal policy
#
# np.savetxt("./20run/discounted_reward_optimal.csv", stats_opt.episode_disc_rewards, delimiter=",")
# np.savetxt("./20run/policy_param_optimal.csv", stats_opt.policy_parameter, delimiter=",")

x = range(num_batch)

mean_alg1 = np.mean(policy_param_off_policy_importance_sampling, axis=0)
mean_alg2 = np.mean(policy_param_off_policy_importance_sampling_pd, axis=0)

var_alg1 = np.std(policy_param_off_policy_importance_sampling, axis=0) / (m.sqrt(runs))
var_alg2 = np.std(policy_param_off_policy_importance_sampling_pd, axis=0) / (m.sqrt(runs))

plot.plot_curves([x, x], [mean_alg1, mean_alg2], [var_alg1, var_alg2], title = "Policy param over batches", x_label = "Batch", y_label = "Policy param", names = ["REINFORCE with transfer (IS)", "REINFORCE with transfer (PD-IS)"])
