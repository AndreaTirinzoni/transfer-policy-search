import gym
import envs
import numpy as np
import algorithmPolicySearch as alg
from matplotlib import pyplot as plt
import importanceWeights as iw

def plot_mean_and_variance_rewards(stats_alg1, stats_alg2, stats_opt, num_batch, discount_factor):
    """
    Plot the mean and standard deviation of the discounted rewards in every batch over the multiple runs
    :param stats_alg1: Set of discounted rewards of alg1 over multiple runs
    :param stats_alg2: Set of discounted rewards of alg2 over multiple runs
    :param stats_opt:  Set of discounted rewards of the optimal policy
    :param num_batch: number of batch in every simulation
    :param discount_factor: discount factor of the algorithm
    :return:
            Returns the plot
    """
    mean_alg1 = np.mean(stats_alg1, axis=0)
    mean_alg2 = np.mean(stats_alg2, axis=0)
    var_alg1 = np.std(stats_alg1, axis=0)
    var_alg2 = np.std(stats_alg1, axis=0)
    x = range(num_batch)

    fig = plt.figure()
    ax= fig.add_subplot ( 111 )

    ax.plot(x, mean_alg1, marker = '.', color = 'red', markersize = 1, linewidth=2, label='REINFORCE')
    ax.plot(x, mean_alg1+var_alg1, marker = '.', color = 'red', markersize = 1, linewidth=0.5, alpha=0.7)
    ax.plot(x, mean_alg1-var_alg1, marker = '.', color = 'red', linewidth=0.5, markersize = 1, alpha=0.7)
    ax.plot(x, mean_alg2, marker = '.', color = 'b', markersize = 1, linewidth=2, label='REINFORCE with transfer')
    ax.plot(x, mean_alg2+var_alg2, marker = '.', color = 'b', markersize = 1, linewidth=0.5, alpha=0.7)
    ax.plot(x, mean_alg2-var_alg2, marker = '.', color = 'b', linewidth=0.5, markersize = 1, alpha=0.7)
    ax.plot(x, stats_opt, marker = '.', color = 'g', linewidth=1, markersize = 1, label='Optimal policy')
    ax.legend()

    title = "Discounted reward over Batches - gamma = " + str(discount_factor)
    plt.title(title)

    # ax.fill(mean_alg2-var_alg2, mean_alg2+var_alg2, 'r', alpha=0.3)
    #
    # # Outline of the region we've filled in
    # ax.plot(mean_alg2-var_alg2, mean_alg2+var_alg2, c='b', alpha=0.8)

    plt.show()

    return fig

def plot_mean_and_variance_policy(stats_alg1, stats_alg2, stats_opt, num_batch, discount_factor):
    """
    Plot the mean and standard deviation of the discounted rewards in every batch over the multiple runs
    :param stats_alg1: Set of discounted rewards of alg1 over multiple runs
    :param stats_alg2: Set of discounted rewards of alg2 over multiple runs
    :param stats_opt:  Set of discounted rewards of the optimal policy
    :param num_batch: number of batch in every simulation
    :param discount_factor: discount factor of the algorithm
    :return:
            Returns the plot
    """
    mean_alg1 = np.mean(stats_alg1, axis=0)
    mean_alg2 = np.mean(stats_alg2, axis=0)
    var_alg1 = np.std(stats_alg1, axis=0)
    var_alg2 = np.std(stats_alg1, axis=0)
    x = range(num_batch)

    fig = plt.figure()
    ax= fig.add_subplot ( 111 )

    ax.plot(x, mean_alg1, marker = '.', color = 'red', markersize = 1, linewidth=2, label='REINFORCE')
    ax.plot(x, mean_alg1+var_alg1, marker = '.', color = 'red', markersize = 1, linewidth=0.5, alpha=0.7)
    ax.plot(x, mean_alg1-var_alg1, marker = '.', color = 'red', linewidth=0.5, markersize = 1, alpha=0.7)
    ax.plot(x, mean_alg2, marker = '.', color = 'b', markersize = 1, linewidth=2, label='REINFORCE with transfer')
    ax.plot(x, mean_alg2+var_alg2, marker = '.', color = 'b', markersize = 1, linewidth=0.5, alpha=0.7)
    ax.plot(x, mean_alg2-var_alg2, marker = '.', color = 'b', linewidth=0.5, markersize = 1, alpha=0.7)
    ax.plot(x, stats_opt, marker = '.', color = 'g', linewidth=1, markersize = 1, label='Optimal policy')
    ax.legend()

    title = "Policy parameters over Batches - gamma = " + str(discount_factor)
    plt.title(title)

    # ax.fill(mean_alg2-var_alg2, mean_alg2+var_alg2, 'r', alpha=0.3)
    #
    # # Outline of the region we've filled in
    # ax.plot(mean_alg2-var_alg2, mean_alg2+var_alg2, c='b', alpha=0.8)

    plt.show()

    return fig

np.set_printoptions(precision=4)
env = gym.make('LQG1D-v0')

mean_initial_param = 0
episode_length = 20
variance_initial_param = 0
variance_action = 0.1
num_episodes = 3000
batch_size = 30
num_batch = num_episodes//batch_size
discount_factor = 0.99
runs = 5

print("Loading files")
# source_task = np.genfromtxt('source_task.csv', delimiter=',')
# episodes_per_config = np.genfromtxt('episodes_per_config.csv', delimiter=',').astype(int)
# source_param = np.genfromtxt('source_param.csv', delimiter=',')
#
# discounted_reward_off_policy = np.zeros((runs, num_batch))
# discounted_reward_reinfroce = np.zeros((runs, num_batch))
# policy_param_off_policy = np.zeros((runs, num_batch))
# policy_param_reinfroce = np.zeros((runs, num_batch))
#
# print("Learn policy")
# for i_run in range(runs):
#     print(i_run)
#     np.random.seed(2000+5*i_run)
#     initial_param = np.random.normal(mean_initial_param, variance_initial_param)
#     off_policy = iw.offPolicyImportanceSampling(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
#     reinforce = alg.reinforce(env, num_episodes, batch_size, discount_factor, episode_length, initial_param, variance_action)
#     discounted_reward_off_policy[i_run,:] = off_policy.episode_disc_rewards
#     discounted_reward_reinfroce[i_run, :] = reinforce.episode_disc_rewards
#     policy_param_off_policy[i_run,:] = off_policy.policy_parameter
#     policy_param_reinfroce[i_run, :] = reinforce.policy_parameter

discounted_reward_off_policy = np.genfromtxt('discounted_reward_off_policy.csv', delimiter=',')
discounted_reward_reinfroce = np.genfromtxt('discounted_reward_reinfroce.csv', delimiter=',')
policy_param_off_policy = np.genfromtxt('policy_param_off_policy.csv', delimiter=',')
policy_param_reinfroce = np.genfromtxt('policy_param_reinfroce.csv', delimiter=',')

stats_opt = iw.optimalPolicy(env, num_episodes, discount_factor, batch_size, episode_length) # Optimal policy

print("Saving files")
np.savetxt("discounted_reward_off_policy.csv", discounted_reward_off_policy, delimiter=",")
np.savetxt("discounted_reward_reinfroce.csv", discounted_reward_reinfroce, delimiter=",")
np.savetxt("policy_param_off_policy.csv", policy_param_off_policy, delimiter=",")
np.savetxt("policy_param_reinfroce.csv", policy_param_reinfroce, delimiter=",")

plot_mean_and_variance_rewards(discounted_reward_reinfroce, discounted_reward_off_policy, stats_opt.episode_disc_rewards, num_batch, discount_factor)
plot_mean_and_variance_policy(policy_param_reinfroce, policy_param_off_policy, stats_opt.policy_parameter, num_batch, discount_factor)
