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
    """
    The function creates a new episode
    """
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

def reinforceAndSourceTaskCreation(env, num_episodes, batch_size, discount_factor, source_task, source_param):
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
    #param = np.random.normal(mean_initial_param, variance_initial_param)
    param = -2
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

                #I populate the source task
                source_task[i_task, t*3] = episode[t, 0]
                source_task[i_task, t*3+1] = episode[t, 1]
                source_task[i_task, t*3+2] = episode[t, 2]

            episode_informations[i_episode,:] = [gradient_est, total_return, discounted_return]

            #I populate the source parameters
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

    gradient_off_policy = np.zeros(num_episodes)

    for i_episode in range(gradient_off_policy.shape[0]):
        gradient_off_policy[i_episode] = sum(source_task[i_episode, 2::3] - param * source_task[i_episode, 1::3] * source_task[i_episode, 1::3] / variance_action)

    return gradient_off_policy

def computeImportanceWeightsSourceTarget(param, source_param, num_episodes, variance_action, source_task, episode_length):

    weights = np.ones(num_episodes)

    for i_episode in range(weights.shape[0]):
        for t in range(episode_length):
            p_src = 1/m.sqrt(2*m.pi*variance_action) * m.exp(-(source_task[i_episode, t*3+1] - param*source_task[i_episode, t*3])**2/(2*variance_action))
            p_tgt = 1/m.sqrt(2*m.pi*variance_action) * m.exp(-(source_task[i_episode, t*3+1] - source_param[i_episode, 1]*source_task[i_episode, t*3])**2/(2*variance_action))
            weights[i_episode] = weights[i_episode] * p_src/p_tgt

    return weights

def offPolicyImportanceSampling(env, num_episodes, batch_size, discount_factor, source_task, source_param, variance_action):

    #param = np.random.normal(mean_initial_param, variance_initial_param)
    param = -2
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Keeps track of useful statistics#
    stats = plot.EpisodeStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch))


    #Compute gradients of the source task
    gradient_off_policy = computeGradients(param, source_task, num_episodes, variance_action)
    #Compute importance weights_source_target of source task
    weights_source_target = computeImportanceWeightsSourceTarget(param, source_param, num_episodes, variance_action, source_task, episode_length)

    for i_batch in range(num_batch):

        episode_informations = np.zeros((batch_size, 3))
        # Create new parameters and new tasks associated to episodes, used tu update the source_param and source_task later
        source_param_new = np.ones((batch_size, 2))
        source_task_new = np.ones((batch_size, episode_length*3))

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
                source_task_new[i_episode, t*3] = episode[t, 0]
                source_task_new[i_episode, t*3+1] = episode[t, 1]
                source_task_new[i_episode, t*3+2] = episode[t, 2]
            episode_informations[i_episode,:] = [gradient_est, total_return, discounted_return]
            source_param_new[i_episode, 0] = discounted_return
            source_param_new[i_episode, 1] = param


        #Update the parameters
        N = weights_source_target.shape[0] + batch_size
        weights_source_target_update = np.concatenate([weights_source_target, np.ones(batch_size)], axis=0) # are the weights used for computing ESS
        gradient_off_policy_update = np.concatenate([gradient_off_policy, episode_informations[:,0]], axis=0)
        discounted_rewards_all = np.concatenate([source_task[:,1], episode_informations[:,2]], axis=0)
        gradient = 1/N * np.dot(np.multiply(weights_source_target_update, gradient_off_policy_update), discounted_rewards_all)
        param, t, m_t, v_t = adam(param, -gradient, t, m_t, v_t, alpha=0.01)

        #Compute EFFECTIVE SAMPLE SIZE
        ess = np.linalg.norm(weights_source_target_update, 1)**2 / np.linalg.norm(weights_source_target_update, 2)**2
        print(ess, N)

        #Compute rewards of batch
        tot_reward_batch = np.mean(episode_informations[:,1])
        discounted_reward_batch = np.mean(episode_informations[:,2])

        # Update weights_source_target, the gradients and source tasks and parameters with new episodes
        gradient_off_policy_new = computeGradients(param, source_task_new, batch_size, variance_action)
        weights_source_target_new = computeImportanceWeightsSourceTarget(param, source_param_new, batch_size, variance_action, source_task_new, episode_length)

        # Concatenate new episodes to source tasks
        source_param = np.concatenate([source_param, source_param_new], axis=0)
        source_task = np.concatenate([source_task, source_task_new], axis=0)
        weights_source_target = np.concatenate([weights_source_target, weights_source_target_new], axis=0)
        gradient_off_policy = np.concatenate([gradient_off_policy, gradient_off_policy_new], axis=0)

        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch
        #print(state, action, reward, param)
    return stats


np.set_printoptions(precision=9)
env1 = gym.make('LQG1D-v0')
env2 = gym.make('LQG1D-v0')
eps = 10**-16
episode_length = 30
mean_initial_param = 0
variance_initial_param = 0.01
variance_action = 0.001
np.random.seed(2000)
num_episodes=800
batch_size=40
num_batch = num_episodes//batch_size
discount_factor = 0.99
source_task = np.zeros((num_episodes, episode_length*3)) # Every line is a task, every task has [discounted_return, policy_parameter]
source_param = np.zeros((num_episodes, 2)) # every line a task, every task has all [state, action, reward]

[stats, source_task, source_param] = reinforceAndSourceTaskCreation(env1, num_episodes, batch_size, discount_factor, source_task, source_param)
stats_off_policy = offPolicyImportanceSampling(env2, num_episodes, batch_size, discount_factor, source_task, source_param, variance_action)
stats_opt = optimalPolicy(env2, num_episodes, batch_size, discount_factor) # Optimal policy

# Compare the statistics on_policy off_policy
plot.plot_algorithm_comparison_total_two(stats, stats_off_policy, stats_opt, num_batch, discount_factor)
