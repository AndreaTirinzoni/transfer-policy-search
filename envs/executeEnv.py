import gym
import itertools
import collections
import numpy as np
import sys
import plotting as plot
env = gym.make('LQG1D-v0')
eps = 10**-16

def reinforce(env, num_episodes, batch_size, discount_factor=1.0):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for
        batch_size: Number of episodes for each batch
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    param = np.zeros(np.shape(env.state))
    # Keeps track of useful statistics#
    stats = plot.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    Episode_values = collections.namedtuple("Episode_values", ["gradient_estimate", "total_reward"])

    # Iterate for all batch
    num_batch = num_episodes//batch_size
    for i_batch in range(num_batch):

        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            # Reset the environment and pick the first action
            state = env.reset()
            episode_informations = []
            episode = []
            total_return = 0
            gradient_est = 0

            # One step in the environment
            for t in range(100):
                #env.render()
                # Take a step
                mean = param*state
                variance = 1
                action = np.random.normal(mean, variance)
                next_state, reward, done, _ = env.step(action)
                # Keep track of the transition
                episode.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

            # Go through the episode and compute estimators
            for t, transition in enumerate(episode):
                # The return after this timestep
                total_return += discount_factor**t * transition.reward
                gradient_est += (transition.action - param * transition.state) * transition.state / variance
            episode_informations.append(Episode_values(gradient_estimate = gradient_est, total_reward = total_return))

        par_old = param
        param = param + 0.01 * 1/batch_size * sum(episode.total_reward * episode.gradient_estimate for t, episode in enumerate(episode_informations))
        if abs(par_old-param) <= eps:
            break

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
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    param = np.zeros(np.shape(env.state))
    # Keeps track of useful statistics#
    stats = plot.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    Episode_values = collections.namedtuple("Episode_values", ["gradient_estimate", "total_reward"])

    # Iterate for all batch
    num_batch = num_episodes//batch_size
    for i_batch in range(num_batch):

        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            # Reset the environment and pick the first action
            state = env.reset()
            episode_informations = []
            episode = []
            total_return = 0
            gradient_est = 0

            # One step in the environment
            for t in range(100):
                #env.render()
                # Take a step
                mean = param*state
                variance = 1
                action = np.random.normal(mean, variance)
                next_state, reward, done, _ = env.step(action)
                # Keep track of the transition
                episode.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

            # Go through the episode and make policy updates
            for t, transition in enumerate(episode):
                # The return after this timestep
                total_return += discount_factor**t * transition.reward
                gradient_est += (transition.action - param * transition.state) * transition.state / variance
            episode_informations.append(Episode_values(gradient_estimate = gradient_est, total_reward = total_return))
        # Compute baseline
        baseline = sum(episode.gradient_estimate**2 * episode.total_reward for t, episode in enumerate(episode_informations))/sum(episode.gradient_estimate**2 for t, episode in enumerate(episode_informations))
        # Update parameters
        par_old = param
        param = param + 0.01 * 1/batch_size * sum(episode.total_reward * episode.gradient_estimate - baseline for t, episode in enumerate(episode_informations))
        if abs(par_old-param) <= eps:
            break

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
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    """
    param = np.zeros(np.shape(env.state))
    # Keeps track of useful statistics#
    stats = plot.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    Episode_values = collections.namedtuple("Episode_values", ["gradient_estimate", "total_reward", "baseline"])

    # Iterate for all batch
    num_batch = num_episodes//batch_size
    for i_batch in range(num_batch):

        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            # Reset the environment and pick the first action
            state = env.reset()
            episode_informations = []
            episode = []
            total_return = 0
            gradient_est = 0

            # One step in the environment
            for t in range(100):
                #env.render()
                # Take a step
                mean = param*state
                variance = 1
                action = np.random.normal(mean, variance)
                next_state, reward, done, _ = env.step(action)
                # Keep track of the transition
                episode.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

            # Go through the episode
            for t, transition in enumerate(episode):
                # The return after this timestep
                total_return += discount_factor**t * transition.reward
                gradient_est += sum((transition.action - param * transition.state) * transition.state / variance for t, transition in enumerate(episode[:t]))
                baseline = 0
                episode_informations.append(Episode_values(gradient_estimate = gradient_est, total_reward = total_return, baseline=baseline)


        par_old = param
        param = param + 0.01 * sum(episode.total_reward * episode.gradient_estimate for t, episode in enumerate(episode_informations))
        if abs(par_old-param) <= eps:
            break

    return stats
"""

def optimalPolicy(env, num_episodes, batch_size):
    """
    Optimal policy (uses Riccati equation)

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    # Keeps track of useful statistics#
    stats_opt = plot.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    # Iterate for all batch
    num_batch = num_episodes//batch_size
    K = env.computeOptimalK()
    for i_batch in range(num_batch):

        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            state = env.reset()
            for t in range(100):
                #env.render()
                # Take a step
                action = K * state
                next_state, reward, done, _ = env.step(action)

                # Update statistics
                stats_opt.episode_rewards[i_episode] += reward
                stats_opt.episode_lengths[i_episode] = t

                if done:
                    break

                state = next_state

    return stats_opt


stats = reinforce(env, 100, 1, discount_factor=0.4)
stats_baseline = reinforceBaseline(env, 100, 1, discount_factor=0.4)
stats_opt = optimalPolicy(env, 100, 1)
#plot the statistics of the algorithm
plot.plot_algorithm_comparison(stats, stats_baseline, stats_opt, 100, smoothing_window=1, discount_factor=0.8)

# Reducing disc_factor baseline improves, after 1 update do I converge to optimal policy?
