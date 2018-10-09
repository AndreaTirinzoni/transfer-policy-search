import gym
import itertools
import math as m
import collections
import numpy as np
import sys
import plotting as plot

def adam(params, grad, t, m_t, v_t, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
    """
    Applies a gradient step to the given parameters based on ADAM update rule
    :param params: a numpy array of parameters
    :param grad: the objective function gradient evaluated in params. This must have the same shape of params
    :param t: the iteration number
    :param m_t: first order momentum
    :param v_t: second order momentum
    :param alpha: base learning rate
    :param beta_1: decay of first order momentum
    :param beta_2: decay of second order momentum
    :param eps: small constant
    :return: the updated parameters, iteration number, first order momentum, and second order momentum
    """

    t += 1
    m_t = beta_1 * m_t + (1 - beta_1) * grad
    v_t = beta_2 * v_t + (1 - beta_2) * grad ** 2
    m_t_hat = m_t / (1 - beta_1 ** t)
    v_t_hat = v_t / (1 - beta_2 ** t)
    return params - alpha * m_t_hat / (np.sqrt(v_t_hat) + eps), t, m_t, v_t

def createEpisode(env, episode_length, param, state, episode):
    for t in range(episode_length):
        #env.render()
        # Take a step
        mean_action = param*state
        action = np.random.normal(mean_action, variance_action)
        next_state, reward, done, _ = env.step(action)
        # Keep track of the transition

        #print(state, action, reward, param)
        episode[t,:] = [state, action, reward, next_state]

        if done:
            break

        state = next_state
    return episode

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
            episode_informations[i_episode,:] = [gradient_est, total_return, discounted_return]

        tot_reward_batch = np.mean(episode_informations[:,1])
        discounted_reward_batch = np.mean(episode_informations[:,2])
        # Update statistics
        stats.episode_total_rewards[i_batch] += tot_reward_batch
        stats.episode_disc_rewards[i_batch] += discounted_reward_batch

        #print(state, action, reward, param)
    return stats

    # Inizialize environment and parameters

def reinforceAndSourceTask(env, num_episodes, batch_size, discount_factor, source_task, source_param):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for
        batch_size: Number of episodes for each batch
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_disc_reward and episode_rewards related to the batch.
    """
    param = np.random.normal(mean_initial_param, variance_initial_param)
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0
    i_task = 0
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
            episode = np.zeros((episode_length, 4)) # [state, action, reward, next_state]
            total_return = 0
            discounted_return = 0
            gradient_est = 0
            episode = createEpisode(env, episode_length, param, state, episode) # [state, action, reward, next_state]

            # Go through the episode and compute estimators
            for t in range(episode.shape[0]):
                # The return after this timestep
                total_return += episode[t, 2]
                discounted_return += discount_factor ** t * episode[t, 2]
                gradient_est += (episode[t, 1] - param * episode[t, 0]) * episode[t, 0] / variance_action
                source_task[i_task, t*3] = episode[t, 0]
                source_task[i_task, t*3+1] = episode[t, 1]
                source_task[i_task, t*3+2] = episode[t, 2]
            episode_informations[i_episode,:] = [gradient_est, total_return, discounted_return]
            source_param[i_task, 0] = discounted_return
            source_param[i_task, 1] = param
            i_task += 1

        gradient = 1/batch_size * np.dot(episode_informations[:,0], episode_informations[:,2])
        param, t, m_t, v_t = adam(param, -gradient, t, m_t, v_t, alpha=0.01)
        tot_reward_batch = np.mean(episode_informations[:,1])
        discounted_reward_batch = np.mean(episode_informations[:,2])
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch
        #print(state, action, reward, param)
    return stats, source_task, source_param

def computeGradients(param, source_task, num_episodes, variance_action):

    gradient_off_policy = np.zeros((num_episodes, 1))

    for i_episode in range(num_episodes):
        gradient_off_policy[i_episode] = sum(source_task[i_episode, 2::3] - param * source_task[i_episode, 1::3] * source_task[i_episode, 1::3] / variance_action)

    return gradient_off_policy

def computeImportanceWeights(param, source_param, num_episodes, variance_action, source_task, episode_length):
    weights = np.zeros((num_episodes, 1))
    p_src = 1
    p_tgt = 1
    for i_episode in range(num_episodes):
        for t in range(episode_length):
            p_src = p_src * 1/m.sqrt(2*m.pi*variance_action) * m.exp(-(source_task[i_episode, t*3] - param*source_task[i_episode, t*3+1])**2/(2*variance_action**2))
            p_tgt = p_tgt * 1/m.sqrt(2*m.pi*variance_action) * m.exp(-(source_task[i_episode, t*3] - source_param[i_episode, 1]*source_task[i_episode, t*3+1])**2/(2*variance_action))
        weights[i_episode] = p_src/p_tgt
    return weights

def offPolicyImportanceSampling(env, num_episodes, batch_size, discount_factor, source_task, source_param, variance_action):

    param = np.random.normal(mean_initial_param, variance_initial_param)
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

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
            episode = np.zeros((episode_length, 4)) # [state, action, reward, next_state]
            total_return = 0
            discounted_return = 0
            gradient_est = 0
            episode = createEpisode(env, episode_length, param, state, episode) # [state, action, reward, next_state]

            # Go through the episode and compute estimators
            for t in range(episode.shape[0]):
                # The return after this timestep
                total_return += episode[t, 2]
                discounted_return += discount_factor ** t * episode[t, 2]
                gradient_est += (episode[t, 1] - param * episode[t, 0]) * episode[t, 0] / variance_action
            episode_informations[i_episode,:] = [gradient_est, total_return, discounted_return]

        #Compute gradients of the source task
        gradient_off_policy = computeGradients(param, source_task, num_episodes, variance_action)
        #Compute importance weights
        weights = computeImportanceWeights(param, source_param, num_episodes, variance_action, source_task, episode_length)

        gradient_source = np.dot(weights * gradient_off_policy, source_task[:,1])
        gradient = 1/batch_size * (np.dot(episode_informations[:,0], episode_informations[:,2]) + gradient_source)
        param, t, m_t, v_t = adam(param, -gradient, t, m_t, v_t, alpha=0.01)
        tot_reward_batch = np.mean(episode_informations[:,1])
        discounted_reward_batch = np.mean(episode_informations[:,2])
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch
        #print(state, action, reward, param)
    return stats


np.set_printoptions(precision=2)
env1 = gym.make('LQG1D-v0')
env2 = gym.make('LQG1D-v0')
eps = 10**-16
episode_length = 100
mean_initial_param = 0
variance_initial_param = 0.01
variance_action = 0.001
np.random.seed(2000)
num_episodes=800
batch_size=40
num_batch = num_episodes//batch_size
discount_factor = 0.99
source_task = np.zeros((num_episodes, episode_length*3))
source_param = np.zeros((num_episodes, 2)) # every line a task, every task has all [state action reward]

[stats, source_task, source_param] = reinforceAndSourceTask(env1, num_episodes, batch_size, discount_factor, source_task, source_param)
stats_off_policy = offPolicyImportanceSampling(env2, num_episodes, batch_size, discount_factor, source_task, source_param, variance_action)
