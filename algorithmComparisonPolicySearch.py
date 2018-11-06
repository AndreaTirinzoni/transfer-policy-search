import gym
import envs
from collections import namedtuple
import numpy as np
import algorithmPolicySearch as alg
import math as m
from utils import plot


def optimalPolicy(env, num_episodes, batch_size, discount_factor, variance_action):
    """
    Optimal policy (uses Riccati equation)
    :param env: OpenAI environment
    :param num_episodes: Number of episodes to run for
    :param discount_factor: the discount factor
    :param batch_size: size of the batch
    :param episode_length: length of each episode
    :return: A BatchStats object with two numpy arrays for episode_disc_reward and episode_rewards
    """

    # Iterate for all batch
    num_batch = num_episodes//batch_size
    # Keeps track of useful statistics#
    stats = BatchStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch),
        policy_parameter=np.zeros(num_batch))
    K = env.computeOptimalK()

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    for i_batch in range(num_batch):

        batch = alg.createBatch(env, batch_size, episode_length, K, variance_action) # [state, action, reward, next_state]

        # The return after this timestep
        total_return = np.sum(batch[:, :, 2], axis=1)
        discounted_return = np.sum((discount_factor_timestep * batch[:, :, 2]), axis=1)

        tot_reward_batch = np.mean(total_return)
        discounted_reward_batch = np.mean(discounted_return)

        # Update statistics
        stats.policy_parameter[i_batch] = K
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch

    return stats

BatchStats = namedtuple("Stats",["episode_total_rewards", "episode_disc_rewards", "policy_parameter"])

# Inizialize environment and parameters
env = gym.make('LQG1D-v0')
episode_length = 20
mean_initial_param = -0.1
variance_initial_param = 0
variance_action = 0.1
num_episodes = 2000
batch_size = 10
num_batch = num_episodes//batch_size
discount_factor = 0.99

runs = 30

reward_reinforce = np.zeros((runs, num_batch))
reward_reinforce_baseline = np.zeros((runs, num_batch))
reward_gpomdp = np.zeros((runs, num_batch))
reward_optimal = np.zeros((runs, num_batch))
policy_reinforce = np.zeros((runs, num_batch))
policy_reinforce_baseline = np.zeros((runs, num_batch))
policy_gpomdp = np.zeros((runs, num_batch))
policy_optimal = np.zeros((runs, num_batch))

print("Learning policy")
for i_run in range(runs):
    # Apply different algorithms to learn optimal policy
    #np.random.seed(2000+5*i_run)
    initial_param = np.random.normal(mean_initial_param, m.sqrt(variance_initial_param))
    print(i_run)

    reinforce = alg.reinforce(env, num_episodes, batch_size, discount_factor, episode_length, initial_param, variance_action) # apply REINFORCE for estimating gradient
    reward_reinforce[i_run,:] = reinforce.episode_disc_rewards
    policy_reinforce[i_run,:] = reinforce.policy_parameter

    reinforce_baseline = alg.reinforceBaseline(env, num_episodes, batch_size, discount_factor, episode_length, initial_param, variance_action) # apply REINFORCE with baseline for estimating gradient
    reward_reinforce_baseline[i_run,:] = reinforce_baseline.episode_disc_rewards
    policy_reinforce_baseline[i_run,:] = reinforce_baseline.policy_parameter

    gpomdp = alg.gpomdp(env, num_episodes, batch_size, discount_factor, episode_length, initial_param, variance_action) # apply G(PO)MDP for estimating gradient
    reward_gpomdp[i_run,:] = gpomdp.episode_disc_rewards
    policy_gpomdp[i_run,:] = gpomdp.policy_parameter

    optimal = optimalPolicy(env, num_episodes, batch_size, discount_factor, variance_action) # Optimal policy
    reward_optimal[i_run,:] = optimal.episode_disc_rewards
    policy_optimal[i_run,:] = optimal.policy_parameter

# Compare the statistics of the different algorithms
mean_alg1 = np.mean(reward_reinforce, axis=0)
mean_alg2 = np.mean(reward_reinforce_baseline, axis=0)
mean_alg3 = np.mean(reward_gpomdp, axis=0)
mean_opt = np.mean(reward_optimal, axis=0)
var_alg1 = np.std(reward_reinforce, axis=0) / (m.sqrt(runs))
var_alg2 = np.std(reward_reinforce_baseline, axis=0) / (m.sqrt(runs))
var_alg3 = np.std(reward_gpomdp, axis=0) / (m.sqrt(runs))
var_opt = np.zeros(num_batch)


mean_pol1 = np.mean(policy_reinforce, axis=0)
mean_pol2 = np.mean(policy_reinforce_baseline, axis=0)
mean_pol3 = np.mean(policy_gpomdp, axis=0)
mean_pol_opt = np.mean(policy_optimal, axis=0)
var_pol1 = np.std(policy_reinforce, axis=0) / (m.sqrt(runs))
var_pol2 = np.std(policy_reinforce_baseline, axis=0) / (m.sqrt(runs))
var_pol3 = np.std(policy_gpomdp, axis=0) / (m.sqrt(runs))
var_opt = np.zeros(num_batch)

x = range(num_batch)

plot.plot_curves([x, x, x, x] , [mean_alg1, mean_alg2, mean_alg3, mean_opt], [var_alg1, var_alg2, var_alg3, var_opt], title = "Rewards over batches", x_label = "Batch", y_label = "Discounted reward", names = ["REINFORCE", "REINFORCE with baseline", "G(PO)MDP", "Optimal policy"], file_name = "Rewards policy search")
plot.plot_curves([x, x, x, x] , [mean_pol1, mean_pol2, mean_pol3, mean_pol_opt], [var_pol1, var_pol2, var_pol3, var_pol_opt], title = "Policy parameter over batches", x_label = "Batch", y_label = "Policy parameter", names = ["REINFORCE", "REINFORCE with baseline", "G(PO)MDP", "Optimal policy"], file_name = "Policy parameter policy search")

print("Saving files")
np.savetxt("discounted_reward_reinforce.csv", reward_reinforce, delimiter=",")
np.savetxt("policy_parameter_reinforce.csv", policy_reinforce, delimiter=",")
np.savetxt("discounted_reward_reinforce_baseline.csv", reward_reinforce_baseline, delimiter=",")
np.savetxt("policy_parameter_reinforce_baseline.csv", policy_reinforce_baseline, delimiter=",")
np.savetxt("discounted_reward_gpomdp.csv", reward_gpomdp, delimiter=",")
np.savetxt("policy_parameter_gpomdp.csv", policy_gpomdp, delimiter=",")
np.savetxt("discounted_reward_optimal.csv", reward_optimal, delimiter=",")
np.savetxt("policy_parameter_optimal.csv", policy_optimal, delimiter=",")
