import gym
from collections import namedtuple
import numpy as np
import plotting as plot
import algorithmPolicySearch as alg

def optimalPolicy(env, num_episodes, batch_size, discount_factor):
    """
    Optimal policy (uses Riccati equation)

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for

    Returns:
        An EpisodeStats object with two numpy arrays for episode_disc_reward and episode_rewards.
    """
    # Iterate for all batch
    num_batch = num_episodes//batch_size
    # Keeps track of useful statistics#
    stats = EpisodeStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch),
        policy_parameter=np.zeros(num_batch))
    K = env.computeOptimalK()
    for i_batch in range(num_batch):
        episode_informations = np.zeros((batch_size, 3))
        stats.policy_parameter[i_batch] = K
        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            state = env.reset()
            episode = np.zeros((episode_length, 4))
            total_return = 0
            discounted_return = 0
            gradient_est = 0

            for t in range(episode_length):
                #env.render()
                # Take a step
                action = K * state
                next_state, reward, done, _ = env.step(action)
                episode[t,:] = [state, action, reward, next_state]

                if done:
                    break

                state = next_state

            for t in range(episode.shape[0]):
                # The return after this timestep
                total_return += episode[t, 2]
                discounted_return += discount_factor ** t * episode[t, 2]
            episode_informations[i_episode,:] = [gradient_est, total_return, discounted_return]

        tot_reward_batch = np.mean(episode_informations[:,1])
        discounted_reward_batch = np.mean(episode_informations[:,2])
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch

        #print(state, action, reward, param)
    return stats

EpisodeStats = namedtuple("Stats",["episode_total_rewards", "episode_disc_rewards", "policy_parameter"])
# Inizialize environment and parameters
env = gym.make('LQG1D-v0')
episode_length = 100
mean_initial_param = -0.1
variance_initial_param = 0.01
num_episodes=2000
batch_size=40
num_batch = num_episodes//batch_size
discount_factor = 0.99

runs = 3
reward_reinforce = np.zeros((runs, num_batch))
reward_reinforce_baseline = np.zeros((runs, num_batch))
reward_gpomdp = np.zeros((runs, num_batch))
policy_reinforce = np.zeros((runs, num_batch))
policy_reinforce_baseline = np.zeros((runs, num_batch))
policy_gpomdp = np.zeros((runs, num_batch))

for i_run in range(runs):
    # Apply different algorithms to learn optimal policy
    np.random.seed(2000+5*i_run)
    initial_param = np.random.normal(mean_initial_param, variance_initial_param)
    print(i_run)

    reinforce = alg.reinforce(env, num_episodes, batch_size, discount_factor, episode_length, initial_param) # apply REINFORCE for estimating gradient
    reward_reinforce[i_run,:] = reinforce.episode_disc_rewards
    policy_reinforce[i_run,:] = reinforce.policy_parameter

    reinforce_baseline = alg.reinforceBaseline(env, num_episodes, batch_size, discount_factor, episode_length, initial_param) # apply REINFORCE with baseline for estimating gradient
    reward_reinforce_baseline[i_run,:] = reinforce_baseline.episode_disc_rewards
    policy_reinforce_baseline[i_run,:] = reinforce_baseline.policy_parameter

    gpomdp = alg.gpomdp(env, num_episodes, batch_size, discount_factor, episode_length, initial_param) # apply G(PO)MDP for estimating gradient
    reward_gpomdp[i_run,:] = gpomdp.episode_disc_rewards
    policy_gpomdp[i_run,:] = gpomdp.policy_parameter

stats_opt = optimalPolicy(env, num_episodes, batch_size, discount_factor) # Optimal policy
# print("REINFORCE")
# print(reward_reinforce)
# print("REINFORCE baseline")
# print(reward_reinforce_baseline)
# print("G(PO)MDP")
# print(gpomdp.episode_disc_rewards)
# print("Optimal")
# print(stats_opt.episode_disc_rewards)

# Compare the statistics of the different algorithms
plot.plot_mean_and_variance(reward_reinforce, reward_reinforce_baseline, reward_gpomdp, stats_opt.episode_disc_rewards, num_batch, discount_factor)
plot.plot_mean_and_variance(policy_reinforce, policy_reinforce_baseline, policy_gpomdp, stats_opt.policy_parameter, num_batch, discount_factor)
