import gym
import itertools
import collections
import numpy as np
import sys
import plotting as plot
env = gym.make('LQG1D-v0')
eps = 10**-16
episode_length = 100
mean_initial_param = 0
variance_initial_param = 0.1

def reinforce(env, num_episodes, batch_size, discount_factor):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for
        batch_size: Number of episodes for each batch
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_disc_reward and episode_rewards.
    """
    param = np.random.normal(mean_initial_param, variance_initial_param)

    # Iterate for all batch
    num_batch = num_episodes//batch_size
    # Keeps track of useful statistics#
    stats = plot.EpisodeStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch))

    for i_batch in range(num_batch):
        episode_informations = np.zeros((batch_size, 3))
        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            # Reset the environment and pick the first action
            state = env.reset()
            episode = np.zeros((episode_length, 4))
            total_return = 0
            discounted_return = 0
            gradient_est = 0

            # One step in the environment
            for t in range(episode_length):
                #env.render()
                # Take a step
                mean_action = param*state
                variance_action = 0.1
                action = np.random.normal(mean_action, variance_action)
                next_state, reward, done, _ = env.step(action)
                # Keep track of the transition

                #print(state, action, reward, param)
                episode[t,:] = [state, action, reward, next_state]

                if done:
                    break

                state = next_state


            # Go through the episode and compute estimators
            for t in range(episode.shape[0]):
                # The return after this timestep
                total_return += episode[t, 2]
                discounted_return += discount_factor ** t * episode[t, 2]
                gradient_est += (episode[t, 1] - param * episode[t, 0]) * episode[t, 0] / variance_action
            episode_informations[t,:] = [gradient_est, total_return, discounted_return]

        param = param + 0.001 * 1/batch_size * np.dot(episode_informations[:,0], episode_informations[:,2])
        tot_reward_batch = np.mean(episode_informations[:,1])
        discounted_reward_batch = np.mean(episode_informations[:,2])
        # Update statistics
        stats.episode_total_rewards[i_batch] += tot_reward_batch
        stats.episode_disc_rewards[i_batch] += discounted_reward_batch
        #print(state, action, reward, param)
    return stats


def reinforceBaseline(env, num_episodes, batch_size, discount_factor):
    """
    REINFORCE with baseline (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for
        batch_size: Number of episodes for each batch
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_disc_reward and episode_rewards.
    """
    param = np.random.normal(mean_initial_param, variance_initial_param)

    # Iterate for all batch
    num_batch = num_episodes//batch_size
    # Keeps track of useful statistics#
    stats = plot.EpisodeStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch))

    for i_batch in range(num_batch):
        episode_informations = np.zeros((batch_size, 3))
        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            # Reset the environment and pick the first action
            state = env.reset()
            episode = np.zeros((episode_length, 4))
            total_return = 0
            discounted_return = 0
            gradient_est = 0

            # One step in the environment
            for t in range(episode_length):
                #env.render()
                # Take a step
                mean_action = param*state
                variance_action = 0.1
                action = np.random.normal(mean_action, variance_action)
                next_state, reward, done, _ = env.step(action)
                # Keep track of the transition

                #print(state, action, reward, param)
                episode[t,:] = [state, action, reward, next_state]

                if done:
                    break

                state = next_state


            # Go through the episode and compute estimators
            for t in range(episode.shape[0]):
                # The return after this timestep
                total_return += episode[t, 2]
                discounted_return += discount_factor ** t * episode[t, 2]
                gradient_est += (episode[t, 1] - param * episode[t, 0]) * episode[t, 0] / variance_action
            episode_informations[t,:] = [gradient_est, total_return, discounted_return]
        baseline = np.dot(episode_informations[:,0]**2, episode_informations[:,2])/sum(episode_informations[:,0]**2)
        # Update parameters
        param = param + 0.001 * 1/batch_size * np.dot(episode_informations[:,0], episode_informations[:,2]-baseline)
        tot_reward_batch = np.mean(episode_informations[:,1])
        discounted_reward_batch = np.mean(episode_informations[:,2])
        # Update statistics
        stats.episode_total_rewards[i_batch] += tot_reward_batch
        stats.episode_disc_rewards[i_batch] += discounted_reward_batch
        #print(state, action, reward, param)
        return stats


def gmdp(env, num_episodes, batch_size, discount_factor=1.0):
    """
    G(PO)MDP (Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for
        batch_size: Number of episodes for each batch
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_disc_reward and episode_rewards.
    """
    """
    param = np.random.normal(mean_initial_param, variance_initial_param)
    # Keeps track of useful statistics#
    stats = plot.EpisodeStats(
        episode_disc_reward=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    Episode_values = collections.namedtuple("Episode_values", ["gradient_estimate", "total_reward", "baseline"])

    # Iterate for all batch
    num_batch = num_episodes//batch_size
    for i_batch in range(num_batch):
        batch_informations = []

        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            # Reset the environment and pick the first action
            state = env.reset()
            episode = []
            total_return = 0
            gradient_est = 0

            # One step in the environment
            for t in range(episode_length):
                #env.render()
                # Take a step
                mean_action = param*state
                variance_action = 1
                action = np.random.normal(mean_action, variance_action)
                next_state, reward, done, _ = env.step(action)
                # Keep track of the transition
                episode.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_disc_reward[i_episode] = t

                if done:
                    break

                state = next_state

            # Go through the episode
            for t, transition in enumerate(episode):
                # The return after this timestep
                total_return += discount_factor**t * transition.reward
                gradient_est += sum((transition.action - param * transition.state) * transition.state / variance_action for t, transition in enumerate(episode[:t]))
                baseline = 0
                batch_informations.append(Episode_values(gradient_estimate = gradient_est, total_reward = total_return, baseline=baseline)

        #estimate = 0

        for t, episode in enumerate(batch_informations):
            estimate = estimate + episode.total_reward * episode.gradient_estimate - baseline

        param = param + 0.01 * update

        if abs(par_old-param) <= eps:
            break

    return stats
"""

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
    stats = plot.EpisodeStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch))
    K = env.computeOptimalK()
    for i_batch in range(num_batch):
        episode_informations = np.zeros((batch_size, 3))
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
            episode_informations[t,:] = [gradient_est, total_return, discounted_return]

        tot_reward_batch = np.mean(episode_informations[:,1])
        discounted_reward_batch = np.mean(episode_informations[:,2])
        # Update statistics
        stats.episode_total_rewards[i_batch] += tot_reward_batch
        stats.episode_disc_rewards[i_batch] += discounted_reward_batch

        #print(state, action, reward, param)
    return stats


num_episodes=1000
batch_size=100
num_batch = num_episodes//batch_size
discount_factor = 0.9
stats = reinforce(env, num_episodes, batch_size, discount_factor)
stats_baseline = reinforceBaseline(env, num_episodes, batch_size, discount_factor)
stats_opt = optimalPolicy(env, num_episodes, batch_size, discount_factor)
print(stats)
print(stats_baseline)
print(stats_opt)
#plot the statistics of the algorithm
plot.plot_algorithm_comparison_total(stats, stats_baseline, stats_opt, num_batch, discount_factor)
#plot.plot_algorithm_comparison_discounted(stats, stats_baseline, stats_opt, num_batch, discount_factor)

# Reducing disc_factor baseline improves, after 1 update do I converge to optimal policy?
