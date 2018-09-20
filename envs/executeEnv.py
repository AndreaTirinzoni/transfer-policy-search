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
        estimator_policy: Policy Function to be optimized
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    param = -np.ones(np.shape(env.state))
    # Keeps track of useful statistics#
    stats_opt = plot.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

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

            K = env.computeOptimalK()

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


            state = env.reset()
            #
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

        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this timestep
            total_return += discount_factor**t * transition.reward
            gradient_est += (transition.action - param * transition.state) * transition.state / variance
            episode_informations.append(Episode_values(gradient_estimate = gradient_est, total_reward = total_return))

        par_old = param
        param = param + 0.01 * 1/batch_size * sum(episode.total_reward * episode.gradient_estimate for t, episode in enumerate(episode_informations))
        if abs(par_old-param) <= eps:
            break

    return stats, stats_opt



stats, stats_opt = reinforce(env, 100, 10, discount_factor=0.6)

#plot the statistics of the algorithm
plot.plot_episode_stats(stats, stats_opt, 100, smoothing_window=1)
#plot.plot_episode_stats(stats_opt, smoothing_window=1)
