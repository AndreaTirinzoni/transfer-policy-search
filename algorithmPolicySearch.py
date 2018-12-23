import numpy as np
from collections import namedtuple
import math as m

class BatchStats:

    def __init__(self, num_batch, param_space_size):

        self.total_rewards = np.zeros(num_batch)
        self.disc_rewards = np.zeros(num_batch)
        self.policy_parameter = np.zeros((num_batch, param_space_size))
        self.gradient = np.zeros((num_batch, param_space_size))
        self.ess = np.zeros(num_batch)

def createBatch(env, batch_size, episode_length, param, state_space_size, variance_action):
    """
    Create a batch of episodes
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param episode_length: length of the episode
    :param param: policy parameter
    :param state_space_size: size of the state space
    :param variance_action: variance of the action's distribution
    :return: A tensor containing [num episodes, timestep, informations] where informations stays for: [state, action, reward, next_state, unclipped_state, unclipped_action]
    """

    information_size = state_space_size+2+state_space_size+state_space_size+1
    batch = np.zeros((batch_size, episode_length, information_size)) #[state, clipped_action, reward, next_state, unclipped_state, action]
    for i_batch in range(batch_size):
        state = env.reset()

        for t in range(episode_length):
            # Take a step
            mean_action = np.dot(param, state)
            action = np.random.normal(mean_action, m.sqrt(variance_action))
            next_state, reward, done, unclipped_state, clipped_action, state_denoised = env.step(action)
            # Keep track of the transition
            #env.render()
            batch[i_batch, t, 0:state_space_size] = state
            batch[i_batch, t, state_space_size] = clipped_action
            batch[i_batch, t, state_space_size+1] = reward
            batch[i_batch, t, state_space_size+2:state_space_size+2+state_space_size] = next_state
            batch[i_batch, t, state_space_size+2+state_space_size:state_space_size+2+state_space_size+state_space_size] = unclipped_state
            batch[i_batch, t, -1] = action

            if done:
                break

            state = next_state

    return batch

def adam(params, grad, t, m_t, v_t, alpha=0.01, beta_1=0.9, beta_2=0.999, eps=1e-8):
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
    :return: The updated parameters, iteration number, first order momentum, and second order momentum
    """

    t += 1
    m_t = beta_1 * m_t + (1 - beta_1) * grad
    v_t = beta_2 * v_t + (1 - beta_2) * grad ** 2
    m_t_hat = m_t / (1 - beta_1 ** t)
    v_t_hat = v_t / (1 - beta_2 ** t)
    return params - alpha * m_t_hat / (np.sqrt(v_t_hat) + eps), t, m_t, v_t, m_t_hat

def reinforce(env, num_batch, batch_size, discount_factor, episode_length, initial_param, variance_action, param_space_size, state_space_size, learning_rate):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy function approximator using ADAM
    :param env: OpenAI environment
    :param num_batch: number of batch
    :param batch_size: number of episodes for each batch
    :param discount_factor: time-discount factor
    :param episode_length: mean initial policy parameter
    :param initial_param: initial policy parameter
    :param variance_action: variance of the action's distribution
    :param param_space_size: size of the parameter space
    :param state_space_size: size of the state space
    :param learning_rate: learning rate of the update rule
    :return: A BatchStats object with two numpy arrays for episode_disc_reward and episode_rewards related to the batch
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Keep track of useful statistics#
    stats = BatchStats(num_batch, param_space_size)

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    for i_batch in range(num_batch):

        stats.policy_parameter[i_batch, :] = param
        #stats.policy_parameter[i_batch] = param #unidimensional policy

        batch = createBatch(env, batch_size, episode_length, param, state_space_size, variance_action) # [state, action, reward, next_state]

        # The return after this timestep
        total_return = np.sum(batch[:, :, state_space_size+1], axis=1)
        discounted_return = np.sum((discount_factor_timestep * batch[:, :, state_space_size+1]), axis=1)
        gradient_est = np.sum((batch[:, :, -1] - np.sum(param[np.newaxis, np.newaxis, :] * batch[:, :, 0:state_space_size], axis=2))[:, :, np.newaxis] * batch[:, :, 0:state_space_size] / variance_action, axis=1)
        episode_informations = np.concatenate([gradient_est, np.asmatrix([total_return, discounted_return]).T], axis=1)
        gradient = np.squeeze(np.asarray(1/batch_size * np.matmul(gradient_est.T, discounted_return)))

        tot_reward_batch = np.mean(episode_informations[:, param_space_size])
        discounted_reward_batch = np.mean(episode_informations[:, -1])
        stats.gradient[i_batch, :] = gradient
        # Update statistics

        #param, t, m_t, v_t, gradient= adam(param, -gradient, t, m_t, v_t)
        param = param + learning_rate * gradient

        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        #print(state, action, reward, param)

    return stats

def reinforceBaseline(env, num_batch, batch_size, discount_factor, episode_length, initial_param, variance_action, param_space_size, state_space_size, learning_rate):
    """
    REINFORCE with BASELINE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy function approximator using ADAM
    :param env: OpenAI environment
    :param num_batch: number of batch
    :param batch_size: number of episodes for each batch
    :param discount_factor: time-discount factor
    :param episode_length: mean initial policy parameter
    :param initial_param: initial policy parameter
    :param variance_action: variance of the action's distribution
    :param param_space_size: size of the parameter space
    :param state_space_size: size of the state space
    :param learning_rate: learning rate of the update rule
    :return: A BatchStats object with two numpy arrays for episode_disc_reward and episode_rewards related to the batch
    """
    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Keep track of useful statistics#

    stats = BatchStats(num_batch, param_space_size)

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    for i_batch in range(num_batch):
        stats.policy_parameter[i_batch, :] = param
        #stats.policy_parameter[i_batch] = param #unidimensional policy

        batch = createBatch(env, batch_size, episode_length, param, state_space_size, variance_action) # [state, action, reward, next_state]

        # The return after this timestep
        total_return = np.sum(batch[:, :, state_space_size+1], axis=1)
        discounted_return = np.sum((discount_factor_timestep * batch[:, :, state_space_size+1]), axis=1)
        gradient_est = np.sum((batch[:, :, -1] - np.sum(param[np.newaxis, np.newaxis, :] *  batch[:, :, 0:state_space_size], axis=2))[:, :, np.newaxis] * batch[:, :, 0:state_space_size] / variance_action, axis=1)

        #Compute gradients
        baseline = np.multiply((gradient_est)**2, discounted_return[:, np.newaxis]) / np.sum(np.squeeze(np.asarray(gradient_est))**2, axis=0)
        gradient = np.squeeze(np.asarray(1/batch_size * np.sum(np.multiply(gradient_est, discounted_return[:, np.newaxis] - baseline), axis=0)))

        tot_reward_batch = np.mean(total_return)
        discounted_reward_batch = np.mean(discounted_return)
        stats.gradient[i_batch, :] = gradient
        # Update statistics

        #param, t, m_t, v_t, gradient= adam(param, -gradient, t, m_t, v_t)
        param = param + learning_rate * gradient

        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        #print(state, action, reward, param)

    return stats

def gpomdp(env, num_batch, batch_size, discount_factor, episode_length, initial_param, variance_action, param_space_size, state_space_size, learning_rate):
    """
    G(PO)MDP (Policy Gradient) Algorithm. Optimizes the policy function approximator using ADAM
    :param env: OpenAI environment
    :param num_batch: number of batch to run for
    :param batch_size: number of episodes for each batch
    :param discount_factor: time-discount factor
    :param episode_length: mean initial policy parameter
    :param initial_param: initial policy parameter
    :param variance_action: variance of the action's distribution
    :param learning_rate: learning rate of the update rule
    :return: A BatchStats object with two numpy arrays for episode_disc_reward and episode_rewards related to the batch
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Keep track of useful statistics#

    stats = BatchStats(num_batch, param_space_size)

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    for i_batch in range(num_batch):
        stats.policy_parameter[i_batch, :] = param

        batch = createBatch(env, batch_size, episode_length, param, state_space_size, variance_action) # [state, action, reward, next_state]

        # The return after this timestep
        total_return = np.sum(batch[:, :, state_space_size+1], axis=1)
        discounted_return = np.sum((discount_factor_timestep * batch[:, :, state_space_size+1]), axis=1)
        gradient_est_timestep = np.cumsum((batch[:, :, -1] - np.sum(param[np.newaxis, np.newaxis, :] * batch[:, :, 0:state_space_size], axis=2))[:, :, np.newaxis] * batch[:, :, 0:state_space_size] / variance_action, axis=1)

        #episode_informations = np.matrix([total_return, discounted_return]).T
        #estimate = 0

        baseline_den = np.sum(gradient_est_timestep**2, axis=0)
        baseline = np.sum((gradient_est_timestep**2) * (discount_factor_timestep * batch[:, :, state_space_size+1])[:, :, np.newaxis], axis=0) / baseline_den

        gradient = 1/batch_size * np.sum(np.sum(gradient_est_timestep * discount_factor_timestep[np.newaxis, :, np.newaxis] * (batch[:, :, state_space_size+1][:, :, np.newaxis] - baseline[np.newaxis, :, :]), axis=1), axis=0)
        # print(baseline, gradient, param)
        #param, t, m_t, v_t, gradient = adam(param, -gradient, t, m_t, v_t)
        param = param + learning_rate * gradient
        tot_reward_batch = np.mean(total_return)
        discounted_reward_batch = np.mean(discounted_return)
        # Update statistics
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        stats.gradient[i_batch, :] = gradient

    return stats
