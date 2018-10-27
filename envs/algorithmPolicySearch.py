import numpy as np
from collections import namedtuple

variance_action = 0.001
EpisodeStats = namedtuple("Stats",["episode_total_rewards", "episode_disc_rewards", "policy_parameter"])

def createEpisode(env, episode_length, param, state):
    """
    The function creates a new episode

    Args:
        env: OpenAI environment
        episode_length: length of the episode
        param: policy parameter
        state: initial state
    """
    episode = np.zeros((episode_length, 4)) # [state, action, reward, next_state]
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


def reinforce(env, num_episodes, batch_size, discount_factor, episode_length, initial_param):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for
        batch_size: Number of episodes for each batch
        discount_factor: Time-discount factor
        mean_initial_param: mean initial policy parameter
        variance_initial_param: variance initial policy parameter

    Returns:
        An EpisodeStats object with two numpy arrays for episode_disc_reward and episode_rewards related to the batch.
    """
    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Iterate for all batch
    num_batch = num_episodes//batch_size
    # Keeps track of useful statistics#
    stats = EpisodeStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch),
        policy_parameter=np.zeros(num_batch))

    for i_batch in range(num_batch):
        episode_informations = np.zeros((batch_size, 3))
        stats.policy_parameter[i_batch] = param
        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            # Reset the environment and pick the first action
            state = env.reset()
            episode = createEpisode(env, episode_length, param, state) # [state, action, reward, next_state]

            # The return after this timestep
            total_return = np.sum(episode[:, 2])
            discounted_return = np.sum(np.multiply(np.power(discount_factor*np.ones(episode.shape[0]), range(episode.shape[0])), episode[:, 2]))
            gradient_est = np.sum(np.multiply((episode[:, 1] - param * episode[:, 0]), episode[:, 0]) / variance_action)
            episode_informations[i_episode,:] = [gradient_est, total_return, discounted_return]


        gradient = 1/batch_size * np.dot(episode_informations[:,0], episode_informations[:,2])
        param, t, m_t, v_t = adam(param, -gradient, t, m_t, v_t, alpha=0.01)
        #param = param + 0.01 * gradient
        tot_reward_batch = np.mean(episode_informations[:,1])
        discounted_reward_batch = np.mean(episode_informations[:,2])
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch
        #print(state, action, reward, param)
    return stats


def reinforceBaseline(env, num_episodes, batch_size, discount_factor, episode_length, initial_param):
    """
    REINFORCE with baseline (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for
        batch_size: Number of episodes for each batch
        discount_factor: Time-discount factor
        mean_initial_param: mean initial policy parameter
        variance_initial_param: variance initial policy parameter

    Returns:
        An EpisodeStats object with two numpy arrays for episode_disc_reward and episode_rewards.
    """
    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Iterate for all batch
    num_batch = num_episodes//batch_size
    # Keeps track of useful statistics#
    stats = EpisodeStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch),
        policy_parameter=np.zeros(num_batch))

    for i_batch in range(num_batch):
        episode_informations = np.zeros((batch_size, 3))
        stats.policy_parameter[i_batch] = param
        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            # Reset the environment and pick the first action
            state = env.reset()

            episode = createEpisode(env, episode_length, param, state) # [state, action, reward, next_state]

            # Go through the episode and compute estimators
            # The return after this timestep
            total_return = np.sum(episode[:, 2])
            discounted_return = np.sum(np.multiply(np.power(discount_factor*np.ones(episode.shape[0]), range(episode.shape[0])), episode[:, 2]))
            gradient_est = np.sum(np.multiply((episode[:, 1] - param * episode[:, 0]), episode[:, 0]) / variance_action)
            episode_informations[i_episode,:] = [gradient_est, total_return, discounted_return]


        baseline = np.dot(episode_informations[:,0]**2, episode_informations[:,2])/np.sum(episode_informations[:,0]**2)
        # baseline = 0
        # Update parameters
        gradient = 1/batch_size * np.dot(episode_informations[:,0], episode_informations[:,2]-baseline)
        param, t, m_t, v_t = adam(param, -gradient, t, m_t, v_t, alpha=0.01)
        #param = param + 0.01 * gradient
        tot_reward_batch = np.mean(episode_informations[:,1])
        discounted_reward_batch = np.mean(episode_informations[:,2])
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch
        #print(state, action, reward, param)
    return stats


def gpomdp(env, num_episodes, batch_size, discount_factor, episode_length, initial_param):
    """
    G(PO)MDP (Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for
        batch_size: Number of episodes for each batch
        discount_factor: Time-discount factor
        mean_initial_param: mean initial policy parameter
        variance_initial_param: variance initial policy parameter

    Returns:
        An EpisodeStats object with two numpy arrays for episode_disc_reward and episode_rewards.
    """
    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Iterate for all batch
    num_batch = num_episodes//batch_size
    # Keeps track of useful statistics#
    stats = EpisodeStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch),
        policy_parameter=np.zeros(num_batch))

    for i_batch in range(num_batch):
        episode_informations = np.zeros((batch_size, 2))
        gradient_est_timestep = np.zeros((batch_size, episode_length))
        reward_est_timestep = np.zeros((batch_size, episode_length))
        stats.policy_parameter[i_batch] = param
        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            # Reset the environment and pick the first action
            state = env.reset()
            total_return = 0
            discounted_return = 0
            gradient_est = 0

            # One episode in the environment
            episode = createEpisode(env, episode_length, param, state) # [state, action, reward, next_state]

            # Go through the episode
            for t, transition in enumerate(episode):
                # The return after this timestep
                total_return += episode[t, 2]
                discounted_return += discount_factor ** t * episode[t, 2]
                gradient_est += (episode[t, 1] - param * episode[t, 0]) * episode[t, 0] / variance_action
                gradient_est_timestep[i_episode, t] = gradient_est
                reward_est_timestep[i_episode, t] = discounted_return
                episode_informations[i_episode,:] = [total_return, discounted_return]
        #estimate = 0

        baseline_den = sum(sum(gradient_est_timestep[i,:]**2) for i in range(batch_size))
        baseline = list(np.dot(gradient_est_timestep[:,i]**2, reward_est_timestep[:,i])/baseline_den for i in range(episode_length))

        gradient = sum(np.dot(gradient_est_timestep[i,:], (reward_est_timestep[i,:]-baseline[:])) for i in range(batch_size))
        # print(baseline, gradient, param)
        param, t, m_t, v_t = adam(param, -gradient, t, m_t, v_t, alpha=0.01)
        #param = param + 0.01 * gradient
        tot_reward_batch = np.mean(episode_informations[:,0])
        discounted_reward_batch = np.mean(episode_informations[:,1])
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch

    return stats
