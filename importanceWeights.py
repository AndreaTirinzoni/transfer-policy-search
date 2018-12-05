import math as m
import numpy as np
from collections import namedtuple
import algorithmPolicySearch as alg
import random

class BatchStats:

    def __init__(self, num_batch):

        self.total_rewards = np.zeros(num_batch)
        self.disc_rewards = np.zeros(num_batch)
        self.policy_parameter = np.zeros(num_batch)
        self.gradient = np.zeros(num_batch)
        self.ess = np.zeros(num_batch)
        self.n_def = np.zeros(num_batch)

def optimalPolicy(env, num_batch, batch_size, discount_factor, variance_action, episode_length):
    """
    Optimal policy (uses Riccati equation)
    :param env: OpenAI environment
    :param num_batch: Number of batch to run for
    :param discount_factor: the discount factor
    :param batch_size: size of the batch
    :param episode_length: length of each episode
    :return: A BatchStats object with two numpy arrays for episode_disc_reward and episode_rewards
    """

    # Keeps track of useful statistics#
    stats = BatchStats(num_batch)
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
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch

    return stats

def createBatch(env, batch_size, episode_length, param, variance_action):
    """
    Create a batch of episodes
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param episode_length: length of the episode
    :param param: policy parameter
    :param variance_action: variance of the action's distribution
    :return: A tensor containing [num episodes, timestep, informations] where informations stays for: [state, action, reward, next_state, unclipped_state, unclipped_action]
    """

    batch = np.zeros((batch_size, episode_length, 6)) # [state, action, reward, next_state, next_state_unclipped, clipped_actions]
    for i_batch in range(batch_size):
        state = env.reset()

        for t in range(episode_length):
            #env.render()
            # Take a step
            mean_action = param*state
            action = np.random.normal(mean_action, m.sqrt(variance_action))
            next_state, reward, done, unclipped_state, clipped_action = env.step(action)
            # Keep track of the transition

            #print(state, action, reward, param)
            batch[i_batch, t, :] = [state, clipped_action, reward, next_state, unclipped_state, action]

            if done:
                break

            state = next_state

    return batch

def computeGradientsSourceTargetTimestep(param, source_task, variance_action):
    """
    Compute the gradients estimation of the source targets with current policy
    :param param: policy parameters
    :param source_task: source tasks [state, action, reward, ...... , reward, state]
    :param variance_action: variance of the action's distribution
    :return: A vector containing all the gradients for each timestep
    """

    state_t = np.delete(source_task[:, 0::3], -1, axis=1)# state t
    action_t = source_task[:, 1::3] # action t

    gradient_off_policy = (action_t - np.asscalar(np.asarray(param)) * state_t) * state_t / variance_action

    return gradient_off_policy

def computeNdef(weights, ess_min, n):
    """
    Compute the number of samples to generare from the target
    :param weights: the weights
    :param ess_min: the minimum effective sample size
    :return:
    """

    w_1 = np.linalg.norm(weights, 1)
    w_2 = np.linalg.norm(weights, 2)
    num_episodes_target = int(max(0, np.ceil((ess_min * w_1 / n) - (w_1 / (w_2 ** 2) * n))))

    return num_episodes_target

#Compute weights of different estimators

def computeImportanceWeightsSourceTarget(policy_param, env_param, source_param, variance_action, source_task, next_states_unclipped, clipped_actions):
    """
    Compute the importance weights considering policy and transition model
    :param policy_param: current policy parameter
    :param env_param: current environment parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :return: Returns the weights of the importance sampling estimator
    """

    param_policy_src = source_param[:, 1][:, np.newaxis] #policy parameter of source
    state_t = np.delete(source_task[:, 0::3], -1, axis=1)# state t
    state_t1 = next_states_unclipped # state t+1
    unclipped_action_t = source_task[:, 1::3] # action t
    variance_env = source_param[:, 4][:, np.newaxis] # variance of the model transition
    A = source_param[:, 2][:, np.newaxis] # environment parameter A of src
    B = source_param[:, 3][:, np.newaxis] # environment parameter B of src

    policy_tgt = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - policy_param * state_t)**2)/(2*variance_action)), axis=1)
    policy_src = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - param_policy_src * state_t)**2)/(2*variance_action)), axis=1)

    model_tgt = np.prod(1/np.sqrt(2*m.pi*variance_env) * np.exp(-((state_t1 - env_param * state_t - B * clipped_actions) **2) / (2*variance_env)), axis=1)
    model_src = np.prod(1/np.sqrt(2*m.pi*variance_env) * np.exp(-((state_t1 - A * state_t - B * clipped_actions) **2) / (2*variance_env)), axis=1)

    weights = policy_tgt / policy_src * model_tgt / model_src

    return weights

def computeImportanceWeightsSourceTargetPerDecision(policy_param, env_param, source_param, variance_action, source_task, next_states_unclipped, clipped_actions):
    """
    Compute the per-decision importance weights considering policy and transition model
    :param policy_param: current policy parameter
    :param env_param: current environment parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :return: Returns the weights of the per-decision importance sampling estimator
    """

    param_policy_src = source_param[:, 1][:, np.newaxis] #policy parameter of source
    state_t = np.delete(source_task[:, 0::3], -1, axis=1)# state t
    state_t1 = next_states_unclipped # state t+1
    unclipped_action_t = source_task[:, 1::3] # action t
    variance_env = source_param[:, 4][:, np.newaxis] # variance of the model transition
    A = source_param[:, 2][:, np.newaxis] # environment parameter A of src
    B = source_param[:, 3][:, np.newaxis] # environment parameter B of src

    policy_tgt = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - policy_param * state_t)**2)/(2*variance_action)), axis=1)
    policy_src = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - param_policy_src * state_t)**2)/(2*variance_action)), axis=1)

    model_tgt = np.cumprod(1/np.sqrt(2*m.pi*variance_env) * np.exp(-((state_t1 - env_param * state_t - B * clipped_actions) **2) / (2*variance_env)), axis=1)
    model_src = np.cumprod(1/np.sqrt(2*m.pi*variance_env) * np.exp(-((state_t1 - A * state_t - B * clipped_actions) **2) / (2*variance_env)), axis=1)

    weights = policy_tgt / policy_src * model_tgt / model_src

    return weights

def computeMultipleImportanceWeightsSourceTarget(policy_param, env_param, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions):
    """
    Compute the multiple importance weights considering policy and transition model
    :param policy_param: current policy parameter
    :param env_param: current environment parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param episodes_per_config: vector containing the number of episodes for each source configuration
    :param src_distributions: source distributions, (the qj of the MIS denominator policy)
    :param batch_size: size of the batch
    :param episode_length: length of the episode
    :return: Returns the weights of the multiple importance sampling estimator and the new qj of the MIS denominator (policy and model)
    """

    n = source_task.shape[0]
    param_indices = np.concatenate(([0], np.cumsum(np.delete(episodes_per_config, -1))))

    param_policy_src = source_param[param_indices, 1][np.newaxis, np.newaxis, :]#policy parameter of source not repeated

    A = source_param[param_indices, 2][np.newaxis, np.newaxis, :] # environment parameter A of src
    B = source_param[param_indices, 3][np.newaxis, np.newaxis, :] # environment parameter B of src

    evaluated_trajectories = src_distributions.shape[0]

    state_t = np.repeat(np.delete(source_task[:, 0::3], -1, axis=1)[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t
    state_t1 = np.repeat(next_states_unclipped[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t+1
    unclipped_action_t = np.repeat(source_task[:, 1::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # action t
    clipped_actions_t = np.repeat(clipped_actions[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # unclipped action t
    variance_env = source_param[:, 4][:, np.newaxis, np.newaxis] # variance of the model transition

    policy_src_new_traj = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[evaluated_trajectories:, :, :] - param_policy_src * state_t[evaluated_trajectories:, :, :])**2)/(2*variance_action)), axis=1)
    model_src_new_traj = np.prod(1/np.sqrt(2*m.pi*variance_env[evaluated_trajectories:, :, :]) * np.exp(-((state_t1[evaluated_trajectories:, :, :] - A * state_t[evaluated_trajectories:, :, :] - B * clipped_actions_t[evaluated_trajectories:, :, :]) **2) / (2*variance_env[evaluated_trajectories:, :, :])), axis=1)

    policy_src_new_param = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[0:evaluated_trajectories, :, 0] - param_policy_src[0:evaluated_trajectories, :, -1] * state_t[0:evaluated_trajectories, :, 0])**2)/(2*variance_action)), axis=1)
    model_src_new_param = np.prod(1/np.sqrt(2*m.pi*variance_env[0:evaluated_trajectories, :, 0]) * np.exp(-((state_t1[0:evaluated_trajectories, :, 0] - A[0:evaluated_trajectories, :, -1] * state_t[0:evaluated_trajectories, :, 0] - B[0:evaluated_trajectories, :, -1] * clipped_actions_t[0:evaluated_trajectories, :, 0]) **2) / (2*variance_env[0:evaluated_trajectories, :, 0])), axis=1)

    policy_tgt = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[:, :, 0] - policy_param*state_t[:, :, 0])**2)/(2*variance_action)), axis = 1)
    model_tgt = np.prod(1/np.sqrt(2*m.pi*variance_env[:, :, 0]) * np.exp(-((state_t1[:, :, 0] - env_param * state_t[:, :, 0] - B[:, :, -1] * clipped_actions_t[:, :, 0]) **2) / (2*variance_env[:, :, 0])), axis = 1)

    if source_task.shape[0]!=src_distributions.shape[0]:
        src_distributions = np.concatenate((src_distributions, np.matrix(policy_src_new_param * model_src_new_param).T), axis=1)
        src_distributions = np.concatenate((src_distributions, np.matrix(policy_src_new_traj * model_src_new_traj)), axis=0)
        mis_denominator = np.squeeze(np.asarray(np.dot(episodes_per_config/n, src_distributions.T).T))
    else:
        mis_denominator = np.squeeze(np.asarray(np.dot(episodes_per_config/n, src_distributions.T).T))

    weights = policy_tgt * model_tgt / mis_denominator

    return [weights, src_distributions]

def computeMultipleImportanceWeightsSourceTargetCv(policy_param, env_param, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, policy_gradients, baseline):
    """
    Compute the multiple importance weights considering policy and transition model
    :param policy_param: current policy parameter
    :param env_param: current environment parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param episodes_per_config: vector containing the number of episodes for each source configuration
    :param src_distributions: source distributions, (the qj of the MIS denominator policy)
    :param batch_size: size of the batch
    :param episode_length: length of the episode
    :param policy_gradients: the gradientf of the policy (nabla log pi)
    :param baseline: 0 the algorithm doesn't require the baseline, 1 if it does
    :return: Returns the weights of the multiple importance sampling estimator and the new qj of the MIS denominator (policy and model) and the control variate
    """

    n = source_task.shape[0]
    param_indices = np.concatenate(([0], np.cumsum(np.delete(episodes_per_config, -1))))

    param_policy_src = source_param[param_indices, 1][np.newaxis, np.newaxis, :]#policy parameter of source not repeated

    A = source_param[param_indices, 2][np.newaxis, np.newaxis, :] # environment parameter A of src
    B = source_param[param_indices, 3][np.newaxis, np.newaxis, :] # environment parameter B of src

    evaluated_trajectories = src_distributions.shape[0]

    state_t = np.repeat(np.delete(source_task[:, 0::3], -1, axis=1)[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t
    state_t1 = np.repeat(next_states_unclipped[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t+1
    unclipped_action_t = np.repeat(source_task[:, 1::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # action t
    clipped_actions_t = np.repeat(clipped_actions[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # unclipped action t
    variance_env = source_param[:, 4][:, np.newaxis, np.newaxis] # variance of the model transition

    policy_src_new_traj = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[evaluated_trajectories:, :, :] - param_policy_src * state_t[evaluated_trajectories:, :, :])**2)/(2*variance_action)), axis=1)
    model_src_new_traj = np.prod(1/np.sqrt(2*m.pi*variance_env[evaluated_trajectories:, :, :]) * np.exp(-((state_t1[evaluated_trajectories:, :, :] - A * state_t[evaluated_trajectories:, :, :] - B * clipped_actions_t[evaluated_trajectories:, :, :]) **2) / (2*variance_env[evaluated_trajectories:, :, :])), axis=1)

    policy_src_new_param = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[0:evaluated_trajectories, :, 0] - param_policy_src[0:evaluated_trajectories, :, -1] * state_t[0:evaluated_trajectories, :, 0])**2)/(2*variance_action)), axis=1)
    model_src_new_param = np.prod(1/np.sqrt(2*m.pi*variance_env[0:evaluated_trajectories, :, 0]) * np.exp(-((state_t1[0:evaluated_trajectories, :, 0] - A[0:evaluated_trajectories, :, -1] * state_t[0:evaluated_trajectories, :, 0] - B[0:evaluated_trajectories, :, -1] * clipped_actions_t[0:evaluated_trajectories, :, 0]) **2) / (2*variance_env[0:evaluated_trajectories, :, 0])), axis=1)

    policy_tgt = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[:, :, 0] - np.asscalar(np.asarray(policy_param))*state_t[:, :, 0])**2)/(2*variance_action)), axis = 1)
    model_tgt = np.prod(1/np.sqrt(2*m.pi*variance_env[:, :, 0]) * np.exp(-((state_t1[:, :, 0] - np.asscalar(np.asarray(env_param)) * state_t[:, :, 0] - B[:, :, -1] * clipped_actions_t[:, :, 0]) **2) / (2*variance_env[:, :, 0])), axis = 1)

    if source_task.shape[0]!=src_distributions.shape[0]:
        src_distributions = np.concatenate((src_distributions, np.matrix(policy_src_new_param * model_src_new_param).T), axis=1)
        src_distributions = np.concatenate((src_distributions, np.matrix(policy_src_new_traj * model_src_new_traj)), axis=0)
        mis_denominator = np.squeeze(np.asarray(np.dot(episodes_per_config/n, src_distributions.T)))
    else:
        mis_denominator = np.squeeze(np.asarray(np.dot(episodes_per_config/n, src_distributions.T)))

    weights = policy_tgt * model_tgt / mis_denominator
    control_variate = (src_distributions / mis_denominator[:, np.newaxis]) - np.ones(src_distributions.shape)

    if baseline == 1:

        baseline_covariate = np.multiply(weights, policy_gradients)[:, np.newaxis]
        control_variate = np.concatenate((control_variate, baseline_covariate), axis=1)

    return [weights, src_distributions, control_variate]

def computeMultipleImportanceWeightsSourceTargetPerDecision(policy_param, env_param, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions):
    """
    Compute the per-decision multiple importance weights considering policy and transition model
    :param policy_param: current policy parameter
    :param env_param: current environment parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param episodes_per_config: vector containing the number of episodes for each source configuration
    :param src_distributions: source distributions, (the qj of the PD-MIS denominator policy)
    :param batch_size: size of the batch
    :param episode_length: length of the episode
    :return: Returns the weights of the per-decision multiple importance sampling estimator and the new qj of the PD-MIS denominator (policy and model)
    """

    n = source_task.shape[0]
    param_indices = np.concatenate(([0], np.cumsum(np.delete(episodes_per_config, -1))))

    param_policy_src = source_param[param_indices, 1][np.newaxis, np.newaxis, :]#policy parameter of source not repeated

    A = source_param[param_indices, 2][np.newaxis, np.newaxis, :] # environment parameter A of src
    B = source_param[param_indices, 3][np.newaxis, np.newaxis, :] # environment parameter B of src

    evaluated_trajectories = src_distributions.shape[0]

    state_t = np.repeat(np.delete(source_task[:, 0::3], -1, axis=1)[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t
    state_t1 = np.repeat(next_states_unclipped[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t+1
    unclipped_action_t = np.repeat(source_task[:, 1::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # action t
    clipped_actions_t = np.repeat(clipped_actions[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # unclipped action t
    variance_env = source_param[:, 4][:, np.newaxis, np.newaxis] # variance of the model transition

    policy_src_new_traj = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[evaluated_trajectories:, :, :] - param_policy_src * state_t[evaluated_trajectories:, :, :])**2)/(2*variance_action)), axis=1)
    model_src_new_traj = np.cumprod(1/np.sqrt(2*m.pi*variance_env[evaluated_trajectories:, :, :]) * np.exp(-((state_t1[evaluated_trajectories:, :, :] - A * state_t[evaluated_trajectories:, :, :] - B * clipped_actions_t[evaluated_trajectories:, :, :]) **2) / (2*variance_env[evaluated_trajectories:, :, :])), axis=1)

    policy_src_new_param = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[0:evaluated_trajectories, :, 0] - param_policy_src[0:evaluated_trajectories, :, -1] * state_t[0:evaluated_trajectories, :, 0])**2)/(2*variance_action)), axis=1)
    model_src_new_param = np.cumprod(1/np.sqrt(2*m.pi*variance_env[0:evaluated_trajectories, :, 0]) * np.exp(-((state_t1[0:evaluated_trajectories, :, 0] - A[0:evaluated_trajectories, :, -1] * state_t[0:evaluated_trajectories, :, 0] - B[0:evaluated_trajectories, :, -1] * clipped_actions_t[0:evaluated_trajectories, :, 0]) **2) / (2*variance_env[0:evaluated_trajectories, :, 0])), axis=1)


    if source_task.shape[0]!=src_distributions.shape[0]:
        src_distributions = np.concatenate((src_distributions, (policy_src_new_param * model_src_new_param)[:, :, np.newaxis]), axis=2)
        src_distributions = np.concatenate((src_distributions, (policy_src_new_traj * model_src_new_traj)), axis=0)

    policy_tgt = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[:, :, 0] - policy_param*state_t[:, :, 0])**2)/(2*variance_action)), axis = 1)
    model_tgt = np.cumprod(1/np.sqrt(2*m.pi*variance_env[:, :, 0]) * np.exp(-((state_t1[:, :, 0] - env_param * state_t[:, :, 0] - (B[:, :, -1] * clipped_actions_t)[:, :, 0]) **2) / (2*variance_env[:, :, 0])), axis = 1)

    mis_denominator = np.sum(episodes_per_config/n * src_distributions, axis=2)

    weights = policy_tgt * model_tgt / mis_denominator

    return [weights, src_distributions]

def computeMultipleImportanceWeightsSourceTargetCvPerDecision(policy_param, env_param, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, policy_gradients, baseline, approximation):
    """
    Compute the per-decision multiple importance weights considering policy and transition model
    :param policy_param: current policy parameter
    :param env_param: current environment parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param episodes_per_config: vector containing the number of episodes for each source configuration
    :param src_distributions: source distributions, (the qj of the PD-MIS denominator policy)
    :param batch_size: size of the batch
    :param episode_length: length of the episode
    :param policy_gradients: the gradientf of the policy (nabla log pi)
    :param baseline: 0 the algorithm doesn't require the baseline, 1 if it does
    :param approximation: 0 the algorithm doesn't use the baseline approximation, 1 if it does
    :return: Returns the weights of the per-decision multiple importance sampling estimator and the new qj of the PD-MIS denominator (policy and model) and the control variate
    """

    n = source_task.shape[0]
    param_indices = np.concatenate(([0], np.cumsum(np.delete(episodes_per_config, -1))))

    param_policy_src = source_param[param_indices, 1][np.newaxis, np.newaxis, :]#policy parameter of source not repeated

    A = source_param[param_indices, 2][np.newaxis, np.newaxis, :] # environment parameter A of src
    B = source_param[param_indices, 3][np.newaxis, np.newaxis, :] # environment parameter B of src

    evaluated_trajectories = src_distributions.shape[0]

    state_t = np.repeat(np.delete(source_task[:, 0::3], -1, axis=1)[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t
    state_t1 = np.repeat(next_states_unclipped[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t+1
    unclipped_action_t = np.repeat(source_task[:, 1::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # action t
    clipped_actions_t = np.repeat(clipped_actions[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # unclipped action t
    variance_env = source_param[:, 4][:, np.newaxis, np.newaxis] # variance of the model transition

    policy_src_new_traj = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[evaluated_trajectories:, :, :] - param_policy_src * state_t[evaluated_trajectories:, :, :])**2)/(2*variance_action)), axis=1)
    model_src_new_traj = np.cumprod(1/np.sqrt(2*m.pi*variance_env[evaluated_trajectories:, :, :]) * np.exp(-((state_t1[evaluated_trajectories:, :, :] - A * state_t[evaluated_trajectories:, :, :] - B * clipped_actions_t[evaluated_trajectories:, :, :]) **2) / (2*variance_env[evaluated_trajectories:, :, :])), axis=1)

    policy_src_new_param = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[0:evaluated_trajectories, :, 0] - param_policy_src[0:evaluated_trajectories, :, -1] * state_t[0:evaluated_trajectories, :, 0])**2)/(2*variance_action)), axis=1)
    model_src_new_param = np.cumprod(1/np.sqrt(2*m.pi*variance_env[0:evaluated_trajectories, :, 0]) * np.exp(-((state_t1[0:evaluated_trajectories, :, 0] - A[0:evaluated_trajectories, :, -1] * state_t[0:evaluated_trajectories, :, 0] - B[0:evaluated_trajectories, :, -1] * clipped_actions_t[0:evaluated_trajectories, :, 0]) **2) / (2*variance_env[0:evaluated_trajectories, :, 0])), axis=1)


    if source_task.shape[0]!=src_distributions.shape[0]:
        src_distributions = np.concatenate((src_distributions, (policy_src_new_param * model_src_new_param)[:, :, np.newaxis]), axis=2)
        src_distributions = np.concatenate((src_distributions, (policy_src_new_traj * model_src_new_traj)), axis=0)

    policy_tgt = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[:, :, 0] - np.asscalar(np.asarray(policy_param))*state_t[:, :, 0])**2)/(2*variance_action)), axis = 1)
    model_tgt = np.cumprod(1/np.sqrt(2*m.pi*variance_env[:, :, 0]) * np.exp(-((state_t1[:, :, 0] - np.asscalar(np.asarray(env_param)) * state_t[:, :, 0] - (B[:, :, -1] * clipped_actions_t)[:, :, 0]) **2) / (2*variance_env[:, :, 0])), axis = 1)

    mis_denominator = np.sum(episodes_per_config/n * src_distributions, axis=2)

    weights = policy_tgt * model_tgt / mis_denominator

    mis_denominator_tensor = np.repeat(mis_denominator[:, :, np.newaxis], param_policy_src.shape[2], axis=2)

    control_variate = (src_distributions / mis_denominator_tensor) - np.ones(src_distributions.shape)

    if baseline == 1:

        if approximation==1:
            baseline_covariate = np.sum(weights * policy_gradients, axis=1)[:, np.newaxis]
            control_variate = np.concatenate((np.sum(control_variate, axis=1), baseline_covariate), axis=1)

        else:
            baseline_covariate = (weights * policy_gradients)[:, :, np.newaxis]
            control_variate = np.concatenate((control_variate, baseline_covariate), axis=2)

    else:
        control_variate = np.sum(control_variate, axis=1)

    return [weights, src_distributions, control_variate]

# Compute the update of the algorithms using different estimators

def addEpisodesToSourceDataset(env, num_episodes_target, episode_length, param, variance_action, discount_factor_timestep, source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, estimator, weights_source_target = None, gradient_off_policy = None):
    """
    Generate a batch of episodes and update the source dataset
    :param env: openAI environment
    :param num_episodes_target: number of episodes of the target environment
    :param episode_length: length of every episode
    :param param: policy param
    :param variance_action: variance of the action's distribution
    :param discount_factor_timestep: discount factor for every timestep gamma^t
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param estimator: the estimator of the current algorithm
    :param weights_source_target: current weights of the source dataset
    :param gradient_off_policy: gradients of the current dataset
    :return: returns the updated dataset and the new informations to update the statistics
    """

    batch_size = num_episodes_target

    source_param_new = np.ones((num_episodes_target, 5))
    source_task_new = np.ones((num_episodes_target, episode_length*3+1))
    # Iterate for every episode in batch

    batch = createBatch(env, batch_size, episode_length, param, variance_action) # [state, action, reward, next_state, next_state_unclipped, clipped_actions]

    # The return after this timestep
    total_return = np.sum(batch[:, :, 2], axis=1)
    discounted_return_timestep = (discount_factor_timestep * batch[:, :, 2])
    discounted_return = np.sum(discounted_return_timestep, axis=1)

    if estimator == "IS":
        gradient_est = np.sum(((batch[:, :, 5] - param * batch[:, :, 0]) * batch[:, :, 0]) / variance_action, axis=1)
        weights_source_target_update = np.concatenate([weights_source_target, np.ones(num_episodes_target)], axis=0) # are the weights used for computing ESS
        #Update the parameters
        weights_source_target_update[np.isnan(weights_source_target_update)] = 0
        gradient_off_policy_update = np.concatenate([gradient_off_policy, np.asarray(gradient_est)], axis=0)
        discounted_rewards_all = np.concatenate([source_param[:,0], np.asarray(discounted_return)], axis=0)

    if estimator == "PD-IS":
        gradient_est = np.cumsum(((batch[:, :, 5] - param * batch[:, :, 0]) * batch[:, :, 0]) / variance_action, axis=1)
        weights_source_target_update = np.concatenate([weights_source_target, np.ones((num_episodes_target, episode_length))], axis=0) # are the weights used for computing ESS
        #Update the parameters
        weights_source_target_update[np.isnan(weights_source_target_update)] = 0
        gradient_off_policy_update = np.concatenate([gradient_off_policy, np.asarray(gradient_est)], axis=0)
        discounted_rewards_all = np.concatenate([source_param[:, 0], np.asmatrix(discounted_return_timestep)], axis=0)

    source_param_new[:, 0] = discounted_return
    source_param_new[:, 1] = param
    source_param_new[:, 2] = env.A
    source_param_new[:, 3] = env.B
    source_param_new[:, 4] = env.sigma_noise**2

    source_task_new[:, 0::3] = np.concatenate((batch[:, :, 0], np.matrix(batch[:, -1, 3]).T), axis=1)
    source_task_new[:, 1::3] = batch[:, :, 5]
    source_task_new[:, 2::3] = batch[:, :, 2]
    next_states_unclipped_new = batch[:, :, 4]
    clipped_actions_new = batch[:, :, 1]

    #Compute rewards of batch
    tot_reward_batch = np.mean(total_return)
    discounted_reward_batch = np.mean(discounted_return)

    # Concatenate new episodes to source tasks
    source_param = np.concatenate((source_param, source_param_new), axis=0)
    source_task = np.concatenate((source_task, source_task_new), axis=0)
    episodes_per_config = np.concatenate((episodes_per_config, [batch_size]))
    next_states_unclipped = np.concatenate((next_states_unclipped, next_states_unclipped_new))
    clipped_actions = np.concatenate((clipped_actions, clipped_actions_new))

    if ((estimator == "IS") | (estimator == "PD-IS")):
        return source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, gradient_off_policy_update, discounted_rewards_all, weights_source_target_update, tot_reward_batch, discounted_reward_batch

    else:
        return source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, tot_reward_batch, discounted_reward_batch

def generateEpisodesAndComputeRewards(env, episode_length, param, variance_action, discount_factor_timestep, source_task, estimator, weights_source_target = None, gradient_off_policy = None):
    """
    Generate episodes to compute statistics if there are no episodes from the target distribution
    :param env: openAI environment
    :param episode_length: length of every episode
    :param param: policy parameter
    :param variance_action: variance of the action's distribution
    :param discount_factor_timestep: discount factor for every timestep gamma^t
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param estimator:
    :param weights_source_target: current weights of the source dataset
    :param gradient_off_policy: gradients of the current dataset
    :return: rewards of the batch
    """
    batch_size = 5
    # Iterate for every episode in batch

    batch = createBatch(env, batch_size, episode_length, param, variance_action) # [state, action, reward, next_state, next_state_unclipped, clipped_actions]

    # The return after this timestep
    total_return = np.sum(batch[:, :, 2], axis=1)
    discounted_return = np.sum((discount_factor_timestep * batch[:, :, 2]), axis=1)

    #Compute rewards of batch
    tot_reward_batch = np.mean(total_return)
    discounted_reward_batch = np.mean(discounted_return)

    if ((estimator == "IS") | (estimator == "PD-IS")):

        weights_source_target_update = weights_source_target # are the weights used for computing ESS
        weights_source_target_update[np.isnan(weights_source_target_update)] = 0
        gradient_off_policy_update = gradient_off_policy

        if estimator == "IS":
            discounted_rewards_all = np.sum(source_task[:, 2::3] * discount_factor_timestep)

        else:
            discounted_rewards_all = source_task[:, 2::3] * discount_factor_timestep
        return tot_reward_batch, discounted_reward_batch, weights_source_target_update, gradient_off_policy_update, discounted_rewards_all

    else:
        return tot_reward_batch, discounted_reward_batch

def offPolicyUpdateImportanceSampling(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, learning_rate, ess_min, adaptive):
    """
    Compute the gradient update of the policy parameter using importance sampling
    :param env: OpenAI environment
    :param param: current policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param batch_size: size of the batch
    :param t: parameter of ADAM
    :param m_t: parameter of ADAM
    :param v_t: parameter of ADAM
    :param discount_factor: the discount factor
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """
    #Compute gradients of the source task
    gradient_off_policy = np.sum(computeGradientsSourceTargetTimestep(param, source_task, variance_action), axis=1)
    #Compute importance weights_source_target of source task
    weights_source_target = computeImportanceWeightsSourceTarget(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions)

    if adaptive == "Yes":
        num_episodes_target = computeNdef(weights_source_target, ess_min, source_task.shape[0])

    else:
        num_episodes_target = batch_size

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    estimator = "IS"

    if num_episodes_target!=0:
        #Generate the episodes and compute the rewards over the batch
        [source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, gradient_off_policy_update, discounted_rewards_all, weights_source_target_update, tot_reward_batch, discounted_reward_batch] = addEpisodesToSourceDataset(env, num_episodes_target, episode_length, param, variance_action, discount_factor_timestep, source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, estimator, weights_source_target, gradient_off_policy)

    else:
        #Generate episodes and compute rewards for estimating rewards (trick)
        [tot_reward_batch, discounted_reward_batch, weights_source_target_update, gradient_off_policy_update, discounted_rewards_all] = generateEpisodesAndComputeRewards(env, episode_length, param, variance_action, discount_factor_timestep, source_param, estimator, weights_source_target, gradient_off_policy)

    ess = np.linalg.norm(weights_source_target_update, 1)**2 / np.linalg.norm(weights_source_target_update, 2)**2
    N = source_task.shape[0]
    gradient = 1/N * np.sum((weights_source_target_update * gradient_off_policy_update) * discounted_rewards_all, axis=0)
    #param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    param = param + learning_rate * gradient

    return source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, num_episodes_target

def offPolicyUpdateImportanceSamplingPerDec(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, learning_rate, ess_min, adaptive):
    """
    Compute the gradient update of the policy parameter using per-decision importance sampling
    :param env: OpenAI environment
    :param param: current policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param batch_size: size of the batch
    :param t: parameter of ADAM
    :param m_t: parameter of ADAM
    :param v_t: parameter of ADAM
    :param discount_factor: the discout factor
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """

    #Compute gradients of the source task
    gradient_off_policy = np.cumsum(computeGradientsSourceTargetTimestep(param, source_task, variance_action), axis=1)
    #Compute importance weights_source_target of source task
    weights_source_target = computeImportanceWeightsSourceTargetPerDecision(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions)

    if adaptive == "Yes":
        num_episodes_target = computeNdef(weights_source_target[:, -1], ess_min, source_task.shape[0])

    else:
        num_episodes_target = batch_size

    discount_factor_timestep = np.power(discount_factor * np.ones(episode_length), range(episode_length))

    estimator = "PD-IS"

    if num_episodes_target!=0:
        #Generate the episodes and compute the rewards over the batch
        [source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, gradient_off_policy_update, discounted_rewards_all, weights_source_target_update, tot_reward_batch, discounted_reward_batch] = addEpisodesToSourceDataset(env, num_episodes_target, episode_length, param, variance_action, discount_factor_timestep, source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, estimator, weights_source_target, gradient_off_policy)

    else:
        #Generate episodes and compute rewards
        [tot_reward_batch, discounted_reward_batch, weights_source_target_update, gradient_off_policy_update, discounted_rewards_all] = generateEpisodesAndComputeRewards(env, episode_length, param, variance_action, discount_factor_timestep, source_param, estimator, weights_source_target, gradient_off_policy)

    ess = np.min(np.linalg.norm(weights_source_target_update, 1, axis=0)**2 / np.linalg.norm(weights_source_target_update, 2, axis=0)**2, axis=0)
    N = source_task.shape[0]
    gradient = 1/N * np.sum(np.sum(weights_source_target_update * gradient_off_policy_update * discounted_rewards_all, axis = 1))
    #param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    param = param + learning_rate * gradient

    return source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, num_episodes_target

def offPolicyUpdateMultipleImportanceSampling(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, learning_rate, ess_min, adaptive):
    """
    Compute the gradient update of the policy parameter using multiple importance sampling
    :param env: OpenAI environment
    :param param: current policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param variance_action: variance of the action's distribution
    :param src_distributions: source distributions, (the qj of the MIS denominator policy)
    :param episode_length: length of the episodes
    :param batch_size: size of the batch
    :param t: parameter of ADAM
    :param m_t: parameter of ADAM
    :param v_t: parameter of ADAM
    :param discount_factor: the discout factor
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """

    #Compute importance weights_source_target of source task

    num_episodes_target = batch_size

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    estimator = "No weights and gradients"

    if num_episodes_target!=0:
        #Generate the episodes and compute the rewards over the batch
        [source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, tot_reward_batch, discounted_reward_batch] = addEpisodesToSourceDataset(env, num_episodes_target, episode_length, param, variance_action, discount_factor_timestep, source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, estimator)

    else:
        #Generate episodes and compute rewards for estimating rewards (trick)
        [tot_reward_batch, discounted_reward_batch] = generateEpisodesAndComputeRewards(env, episode_length, param, variance_action, discount_factor_timestep, source_param, estimator)

    #Update the parameters
    N = source_task.shape[0]
    [weights_source_target_update, src_distributions] = computeMultipleImportanceWeightsSourceTarget(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions)
    weights_source_target_update[np.isnan(weights_source_target_update)] = 0

    gradient_off_policy_update = np.sum(computeGradientsSourceTargetTimestep(param, source_task, variance_action), axis=1)
    discounted_rewards_all = source_param[:,0]
    gradient = 1/N * np.sum((np.squeeze(np.array(weights_source_target_update)) * gradient_off_policy_update) * discounted_rewards_all)
    #param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    param = param + learning_rate * gradient

    weights_source_target_ess = computeMultipleImportanceWeightsSourceTarget(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions)[0]
    weights_source_target_ess[np.isnan(weights_source_target_ess)] = 0
    ess = np.linalg.norm(weights_source_target_ess, 1)**2 / np.linalg.norm(weights_source_target_ess, 2)**2

    if adaptive=="Yes":
        #Number of n_def next iteration
        num_episodes_target = computeNdef(weights_source_target_ess, ess_min, source_task.shape[0])

    else:
        num_episodes_target = batch_size

    return source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions, num_episodes_target

def regressionFitting(y, x, n_config_cv):
    """
    Fit a regression with the control variates
    :param y: the target
    :param x: the parameters
    :param n_config_cv: number of control variates to fit
    :return: returns the error of the fitted regression
    """
    baseline = np.squeeze(np.asarray(x[:, -1]))[:, np.newaxis]
    x = np.concatenate([x[:, 0:n_config_cv], baseline], axis=1)

    train_size = int(np.ceil(x.shape[0]/3*2))
    train_index = random.sample(range(x.shape[0]), train_size)
    x_train = x[train_index]
    y_train = y[train_index]
    x_test = np.delete(x, train_index, axis=0)
    y_test = np.delete(y, train_index, axis=0)
    beta = np.squeeze(np.asarray(np.matmul(np.linalg.pinv(x_train[:, 1:]), y_train)))
    error = np.squeeze(np.asarray(y_test - np.dot(x_test[:, 1:], beta)))
    # x_avg = np.squeeze(np.asarray(np.mean(x, axis=0)))
    # beta = np.matmul(np.linalg.inv(np.matmul((x[:, 1:]-x_avg[1:]).T, (x[:, 1:]-x_avg[1:]))), np.matmul((x[:, 1:]-x_avg[1:]).T, (y-y_avg)).T)
    # I = np.identity(y.shape[0])
    # error = np.squeeze(np.asarray(np.matmul(I - np.matmul(x[:, 1:], np.linalg.pinv(x[:, 1:])), y)))

    return np.mean(error)

def offPolicyUpdateMultipleImportanceSamplingCv(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, baseline, learning_rate, ess_min, adaptive, n_config_cv):
    """
    Compute the gradient update of the policy parameter using multiple importance sampling
    :param env: OpenAI environment
    :param param: current policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param variance_action: variance of the action's distribution
    :param src_distributions: source distributions, (the qj of the MIS denominator policy)
    :param episode_length: length of the episodes
    :param batch_size: size of the batch
    :param t: parameter of ADAM
    :param m_t: parameter of ADAM
    :param v_t: parameter of ADAM
    :param discount_factor: the discout factor
    :param baseline: 0 the algorithms doesn't have the baseline, 1 it has
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """

    #Compute importance weights_source_target of source task
    num_episodes_target = batch_size

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    estimator = "No weights and gradients"

    if num_episodes_target != 0:
        #Generate the episodes and compute the rewards over the batch
        [source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, tot_reward_batch, discounted_reward_batch] = addEpisodesToSourceDataset(env, num_episodes_target, episode_length, param, variance_action, discount_factor_timestep, source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, estimator)

    else:
        #Generate episodes and compute rewards for estimating rewards (trick)
        [tot_reward_batch, discounted_reward_batch] = generateEpisodesAndComputeRewards(env, episode_length, param, variance_action, discount_factor_timestep, source_param, estimator)


    #Update the parameters
    N = source_task.shape[0]
    gradient_off_policy_update = np.sum(computeGradientsSourceTargetTimestep(param, source_task, variance_action), axis=1)
    [weights_source_target_update, src_distributions, control_variates] = computeMultipleImportanceWeightsSourceTargetCv(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, gradient_off_policy_update, baseline)
    weights_source_target_update[np.isnan(weights_source_target_update)] = 0

    discounted_rewards_all = source_param[:,0]
    gradient_estimation = (np.squeeze(np.array(weights_source_target_update)) * gradient_off_policy_update) * discounted_rewards_all

    #Fitting the regression
    gradient = regressionFitting(gradient_estimation, control_variates, n_config_cv)

    #Update the parameter
    #param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    param = param + learning_rate * gradient

    weights_source_target_ess = computeMultipleImportanceWeightsSourceTarget(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions)[0]
    weights_source_target_ess[np.isnan(weights_source_target_ess)] = 0
    ess = np.linalg.norm(weights_source_target_ess, 1)**2 / np.linalg.norm(weights_source_target_ess, 2)**2

    if adaptive == "Yes":
        #Number of n_def next iteration
        num_episodes_target = computeNdef(weights_source_target_ess, ess_min, source_task.shape[0])

    else:
        num_episodes_target = batch_size

    return source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions, num_episodes_target

def offPolicyUpdateMultipleImportanceSamplingPerDec(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, learning_rate, ess_min, adaptive):
    """
    Compute the gradient update of the policy parameter using per decision multiple importance sampling
    :param env: OpenAI environment
    :param param: current policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param variance_action: variance of the action's distribution
    :param src_distributions: source distributions, (the qj of the PD-MIS denominator policy)
    :param episode_length: length of the episodes
    :param batch_size: size of the batch
    :param t: parameter of ADAM
    :param m_t: parameter of ADAM
    :param v_t: parameter of ADAM
    :param discount_factor: the discout factor
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """

    #Compute gradients of the source task
    num_episodes_target = batch_size

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    estimator = "No weights and gradients"

    if num_episodes_target != 0:
        #Generate the episodes and compute the rewards over the batch
        [source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, tot_reward_batch, discounted_reward_batch] = addEpisodesToSourceDataset(env, num_episodes_target, episode_length, param, variance_action, discount_factor_timestep, source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, estimator)

    else:
        #Generate episodes and compute rewards for estimating rewards (trick)
        [tot_reward_batch, discounted_reward_batch] = generateEpisodesAndComputeRewards(env, episode_length, param, variance_action, discount_factor_timestep, source_param, estimator)


    #Update the parameters
    N = source_task.shape[0]
    #Compute importance weights_source_target of source task
    [weights_source_target_update, src_distributions] = computeMultipleImportanceWeightsSourceTargetPerDecision(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions)
    weights_source_target_update[np.isnan(weights_source_target_update)] = 0

    gradient_off_policy_update = np.cumsum(computeGradientsSourceTargetTimestep(param, source_task, variance_action), axis=1)
    discounted_rewards_all = discount_factor_timestep * source_task[:, 2::3]
    gradient = 1/N * np.sum(np.sum((weights_source_target_update * gradient_off_policy_update) * discounted_rewards_all, axis = 1), axis=0)
    #param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    param = param + learning_rate * gradient

    weights_source_target_ess = computeMultipleImportanceWeightsSourceTargetPerDecision(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions)[0]
    weights_source_target_ess[np.isnan(weights_source_target_ess)] = 0
    ess = np.min(np.linalg.norm(weights_source_target_ess, 1, axis=0)**2 / np.linalg.norm(weights_source_target_ess, 2, axis=0)**2, axis=0)
    min_index = np.argmin(np.linalg.norm(weights_source_target_ess, 1, axis=0)**2 / np.linalg.norm(weights_source_target_ess, 2, axis=0)**2, axis=0)

    if adaptive=="Yes":
        #Number of n_def next iteration
        num_episodes_target = computeNdef(weights_source_target_ess[:, min_index], ess_min, source_task.shape[0])

    else:
        num_episodes_target = batch_size

    return source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions, num_episodes_target

def offPolicyUpdateMultipleImportanceSamplingCvPerDec(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, baseline, approximation, learning_rate, ess_min, adaptive, n_config_cv):
    """
    Compute the gradient update of the policy parameter using per decision multiple importance sampling
    :param env: OpenAI environment
    :param param: current policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param variance_action: variance of the action's distribution
    :param src_distributions: source distributions, (the qj of the PD-MIS denominator policy)
    :param episode_length: length of the episodes
    :param batch_size: size of the batch
    :param t: parameter of ADAM
    :param m_t: parameter of ADAM
    :param v_t: parameter of ADAM
    :param discount_factor: the discout factor
    :param baseline: 0 the algorithms doesn't have the baseline, 1 it has
    :param approximation: 0 the algorithms doesn't use the baseline approximation, 1 it does
    :param learning_rate: learning rate of the update rule
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """

    #Compute gradients of the source task
    num_episodes_target = batch_size

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    estimator = "No weights and gradients"

    if num_episodes_target != 0:
        #Generate the episodes and compute the rewards over the batch
        [source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, tot_reward_batch, discounted_reward_batch] = addEpisodesToSourceDataset(env, num_episodes_target, episode_length, param, variance_action, discount_factor_timestep, source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, estimator)

    else:
        #Generate episodes and compute rewards for estimating rewards (trick)
        [tot_reward_batch, discounted_reward_batch] = generateEpisodesAndComputeRewards(env, episode_length, param, variance_action, discount_factor_timestep, source_param, estimator)

    #Update the parameters
    N = source_task.shape[0]
    #Compute importance weights_source_target of source task
    gradient_off_policy_update = np.cumsum(computeGradientsSourceTargetTimestep(param, source_task, variance_action), axis=1)
    [weights_source_target_update, src_distributions, control_variates] = computeMultipleImportanceWeightsSourceTargetCvPerDecision(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, gradient_off_policy_update, baseline, approximation)
    weights_source_target_update[np.isnan(weights_source_target_update)] = 0

    discounted_rewards_all = discount_factor_timestep * source_task[:, 2::3]
    gradient_estimation = np.sum((weights_source_target_update * gradient_off_policy_update) * discounted_rewards_all, axis=1)

    #Fitting the regression
    gradient = regressionFitting(gradient_estimation, control_variates, n_config_cv) #always the same, only the MIS with CV changes format of the x_avg array

    #Update the parameter
    #param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    param = param + learning_rate * gradient

    weights_source_target_ess = computeMultipleImportanceWeightsSourceTargetPerDecision(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions)[0]
    weights_source_target_ess[np.isnan(weights_source_target_ess)] = 0
    ess = np.min(np.linalg.norm(weights_source_target_ess, 1, axis=0)**2 / np.linalg.norm(weights_source_target_ess, 2, axis=0)**2, axis=0)
    min_index = np.argmin(np.linalg.norm(weights_source_target_ess, 1, axis=0)**2 / np.linalg.norm(weights_source_target_ess, 2, axis=0)**2, axis=0)

    if adaptive=="Yes":
        #Number of n_def next iteration
        num_episodes_target = computeNdef(weights_source_target_ess[:, min_index], ess_min, source_task.shape[0])

    else:
        num_episodes_target = batch_size

    return source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions, num_episodes_target

def offPolicyUpdateMultipleImportanceSamplingCvPerDecBaseline(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, baseline, approximation, learning_rate, ess_min, adaptive, n_config_cv):
    """
    Compute the gradient update of the policy parameter using per decision multiple importance sampling
    :param env: OpenAI environment
    :param param: current policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param variance_action: variance of the action's distribution
    :param src_distributions: source distributions, (the qj of the PD-MIS denominator policy)
    :param episode_length: length of the episodes
    :param batch_size: size of the batch
    :param t: parameter of ADAM
    :param m_t: parameter of ADAM
    :param v_t: parameter of ADAM
    :param discount_factor: the discout factor
    :param baseline: 0 the algorithms doesn't have the baseline, 1 it has
    :param approximation: 0 the algorithms doesn't use the baseline approximation, 1 it does
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """

    #Compute gradients of the source task
    num_episodes_target = batch_size

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    estimator = "No weights and gradients"

    if num_episodes_target != 0:
        #Generate the episodes and compute the rewards over the batch
        [source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, tot_reward_batch, discounted_reward_batch] = addEpisodesToSourceDataset(env, num_episodes_target, episode_length, param, variance_action, discount_factor_timestep, source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, estimator)

    else:
        #Generate episodes and compute rewards
        [tot_reward_batch, discounted_reward_batch] = generateEpisodesAndComputeRewards(env, episode_length, param, variance_action, discount_factor_timestep, source_param, estimator)

    #Update the parameters
    N =  source_task.shape[0]
    #Compute importance weights_source_target of source task
    gradient_off_policy_update = np.cumsum(computeGradientsSourceTargetTimestep(param, source_task, variance_action), axis=1)
    [weights_source_target_update, src_distributions, control_variates] = computeMultipleImportanceWeightsSourceTargetCvPerDecision(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, gradient_off_policy_update, baseline, approximation)
    weights_source_target_update[np.isnan(weights_source_target_update)] = 0

    discounted_rewards_all = discount_factor_timestep * source_task[:, 2::3]
    gradient_estimation = (weights_source_target_update * gradient_off_policy_update) * discounted_rewards_all
    gradient = 0

    #Fitting the regression for every t 0...T-1
    for t in range(control_variates.shape[1]):
        gradient += regressionFitting(gradient_estimation[:, t], control_variates[:, t, :], n_config_cv) #always the same, only the MIS with CV changes format of the x_avg array

    #Update the parameter
    #param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    param = param + learning_rate * gradient

    weights_source_target_ess = computeMultipleImportanceWeightsSourceTargetPerDecision(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions)[0]
    weights_source_target_ess[np.isnan(weights_source_target_ess)] = 0
    ess = np.min(np.linalg.norm(weights_source_target_ess, 1, axis=0)**2 / np.linalg.norm(weights_source_target_ess, 2, axis=0)**2, axis=0)
    min_index = np.argmin(np.linalg.norm(weights_source_target_ess, 1, axis=0)**2 / np.linalg.norm(weights_source_target_ess, 2, axis=0)**2, axis=0)

    if adaptive=="Yes":
        #Number of n_def next iteration
        num_episodes_target = computeNdef(weights_source_target_ess[:, min_index], ess_min, source_task.shape[0])

    else:
        num_episodes_target = batch_size

    return source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions, num_episodes_target

# Algorithm off policy using different estimators

def offPolicyImportanceSampling(env, batch_size, discount_factor, source_task, next_states_unclipped, clipped_actions, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, learning_rate, ess_min, adaptive):
    """
    Perform transfer from source tasks, using REINFORCE with IS
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Keep track of useful statistics#
    stats = BatchStats(num_batch)

    for i_batch in range(num_batch):

        [source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, n_def] = offPolicyUpdateImportanceSampling(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, learning_rate, ess_min, adaptive)

        # Update statistics
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        stats.policy_parameter[i_batch] = param
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess
        stats.n_def[i_batch] = n_def

    return stats

def offPolicyImportanceSamplingPd(env, batch_size, discount_factor, source_task, next_states_unclipped, clipped_actions, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, learning_rate, ess_min, adaptive):
    """
    Perform transfer from source tasks, using REINFORCE with PD-IS
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Keep track of useful statistics#
    stats = BatchStats(num_batch)

    for i_batch in range(num_batch):

        [source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, n_def] = offPolicyUpdateImportanceSamplingPerDec(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, learning_rate, ess_min, adaptive)

        # Update statistics
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        stats.policy_parameter[i_batch] = param
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess
        stats.n_def[i_batch] = n_def

    return stats

def computeMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config):
    """
    Compute the qj of the MIS denominator
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :return: a matrix (N trajectories * M parameters) containing the qj related to the denominator of the MIS weights
    """

    param_indices = np.concatenate(([0], np.cumsum(np.delete(episodes_per_config, -1))))

    param_policy_src = source_param[param_indices, 1][np.newaxis, np.newaxis, :]#policy parameter of source not repeated
    variance_env = source_param[:, 4][:, np.newaxis, np.newaxis] # variance of the model transition
    A = source_param[param_indices, 2][np.newaxis, np.newaxis, :] # environment parameter A of src
    B = source_param[param_indices, 3][np.newaxis, np.newaxis, :] # environment parameter B of src

    state_t = np.repeat(np.delete(source_task[:, 0::3], -1, axis=1)[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t
    state_t1 = np.repeat(next_states_unclipped[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t+1
    unclipped_action_t = np.repeat(source_task[:, 1::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # action t
    clipped_actions_t = np.repeat(clipped_actions[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # action t

    src_distributions_policy = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - param_policy_src * state_t)**2)/(2*variance_action)), axis=1)

    src_distributions_model = np.prod(1/np.sqrt(2*m.pi*variance_env) * np.exp(-((state_t1 - A * state_t - B * clipped_actions_t) **2) / (2*variance_env)), axis=1)

    return src_distributions_model * src_distributions_policy

def computeInitialNdefMultipleImportanceWeights(policy_param, env_param, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, ess_min):
    """
    Compute the multiple importance weights considering policy and transition model
    :param policy_param: current policy parameter
    :param env_param: current environment parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param episodes_per_config: vector containing the number of episodes for each source configuration
    :param src_distributions: source distributions, (the qj of the MIS denominator policy)
    :param batch_size: size of the batch
    :param episode_length: length of the episode
    :param ess_min: minimum effective sample size
    :return: Returns the weights of the multiple importance sampling estimator and the new qj of the MIS denominator (policy and model)
    """

    weights = computeMultipleImportanceWeightsSourceTarget(policy_param, env_param, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions)[0]

    num_episodes_target = computeNdef(weights, ess_min, source_task.shape[0])

    return num_episodes_target

def offPolicyMultipleImportanceSampling(env, batch_size, discount_factor, source_task, next_states_unclipped, clipped_actions, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, learning_rate, ess_min, adaptive):
    """
    Perform transfer from source tasks, using REINFORCE with MIS
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    src_distributions = computeMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config)

    # Keep track of useful statistics#
    stats = BatchStats(num_batch)

    if adaptive=="Yes":
        #Number of n_def next iteration
        n_def = computeInitialNdefMultipleImportanceWeights(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, ess_min)

    else:
        n_def = batch_size

    for i_batch in range(num_batch):

        stats.n_def[i_batch] = n_def
        batch_size = n_def

        [source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions, n_def] = offPolicyUpdateMultipleImportanceSampling(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, learning_rate, ess_min, adaptive)
        # Update statistics
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        stats.policy_parameter[i_batch] = param
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess

    return stats

def offPolicyMultipleImportanceSamplingCv(env, batch_size, discount_factor, source_task, next_states_unclipped, clipped_actions, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, learning_rate, ess_min, adaptive, n_config_cv):
    """
    Perform transfer from source tasks, using REINFORCE with MIS anc the Control Variate
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0
    baseline = 0

    src_distributions = computeMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config)

    # Keep track of useful statistics#
    stats = BatchStats(num_batch)

    if adaptive=="Yes":
        #Number of n_def next iteration
        n_def = computeInitialNdefMultipleImportanceWeights(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, ess_min)

    else:
        n_def = batch_size

    for i_batch in range(num_batch):

        stats.n_def[i_batch] = n_def
        batch_size = n_def

        [source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions, n_def] = offPolicyUpdateMultipleImportanceSamplingCv(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, baseline, learning_rate, ess_min, adaptive, n_config_cv)

        # Update statistics
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        stats.policy_parameter[i_batch] = param
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess

    return stats

def offPolicyMultipleImportanceSamplingCvBaseline(env, batch_size, discount_factor, source_task, next_states_unclipped, clipped_actions, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, learning_rate, ess_min, adaptive, n_config_cv):
    """
    Perform transfer from source tasks, using REINFORCE with MIS anc the Control Variate with baseline
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0
    baseline = 1

    src_distributions = computeMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config)

    # Keep track of useful statistics#
    stats = BatchStats(num_batch)

    if adaptive=="Yes":
        #Number of n_def next iteration
        n_def = computeInitialNdefMultipleImportanceWeights(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, ess_min)

    else:
        n_def = batch_size

    for i_batch in range(num_batch):

        stats.n_def[i_batch] = n_def
        batch_size = n_def

        [source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions, n_def] = offPolicyUpdateMultipleImportanceSamplingCv(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, baseline, learning_rate, ess_min, adaptive, n_config_cv)

        # Update statistics
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        stats.policy_parameter[i_batch] = param
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess

    return stats

def computePerDecisionMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config):
    """
    Compute the qj of the PD-MIS denominator
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :return: a tensor (N trajectories * T timesteps * M parameters) containing the qj related to the denominator of the PD-MIS weights
    """

    param_indices = np.concatenate(([0], np.cumsum(np.delete(episodes_per_config, -1))))

    param_policy_src = source_param[param_indices, 1][np.newaxis, np.newaxis, :]#policy parameter of source not repeated
    variance_env = source_param[:, 4][:, np.newaxis, np.newaxis]# variance of the model transition
    A = source_param[param_indices, 2][np.newaxis, np.newaxis, :] # environment parameter A of src
    B = source_param[param_indices, 3][np.newaxis, np.newaxis, :] # environment parameter B of src

    state_t = np.repeat(np.delete(source_task[:, 0::3], -1, axis=1)[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t
    state_t1 = np.repeat(next_states_unclipped[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t+1
    unclipped_action_t = np.repeat(source_task[:, 1::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # action t
    clipped_actions_t = np.repeat(clipped_actions[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # action t

    src_distributions_policy = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - param_policy_src * state_t)**2)/(2*variance_action)), axis=1)

    src_distributions_model = np.cumprod(1/np.sqrt(2*m.pi*variance_env) * np.exp(-((state_t1 - A * state_t - B * clipped_actions_t) **2) / (2*variance_env)), axis=1)

    return src_distributions_model * src_distributions_policy

def computeInitialNdefPerDecisionMultipleImportanceWeights(policy_param, env_param, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, ess_min):
    """
    Compute the multiple importance weights considering policy and transition model
    :param policy_param: current policy parameter
    :param env_param: current environment parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param episodes_per_config: vector containing the number of episodes for each source configuration
    :param src_distributions: source distributions, (the qj of the MIS denominator policy)
    :param batch_size: size of the batch
    :param episode_length: length of the episode
    :return: Returns the weights of the multiple importance sampling estimator and the new qj of the MIS denominator (policy and model)
    """

    weights = computeMultipleImportanceWeightsSourceTargetPerDecision(policy_param, env_param, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions)[0]
    min_index = np.argmin(np.linalg.norm(weights, 1, axis=0)**2 / np.linalg.norm(weights, 2, axis=0)**2, axis=0)

    num_episodes_target = computeNdef(weights[:, min_index], ess_min, source_task.shape[0])

    return num_episodes_target

def offPolicyMultipleImportanceSamplingPd(env, batch_size, discount_factor, source_task, next_states_unclipped, clipped_actions, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, learning_rate, ess_min, adaptive):
    """
    Perform transfer from source tasks, using REINFORCE with PD-MIS
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    src_distributions = computePerDecisionMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config)

    # Keep track of useful statistics#
    stats = BatchStats(num_batch)

    if adaptive=="Yes":
        #Number of n_def next iteration
        n_def = computeInitialNdefPerDecisionMultipleImportanceWeights(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, ess_min)

    else:
        n_def = batch_size

    for i_batch in range(num_batch):

        stats.n_def[i_batch] = n_def
        batch_size = n_def

        [source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions, n_def] = offPolicyUpdateMultipleImportanceSamplingPerDec(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, learning_rate, ess_min, adaptive)

        # Update statistics
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        stats.policy_parameter[i_batch] = param
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess

    return stats

def offPolicyMultipleImportanceSamplingCvPd(env, batch_size, discount_factor, source_task, next_states_unclipped, clipped_actions, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, learning_rate, ess_min, adaptive, n_config_cv):
    """
    Perform transfer from source tasks, using REINFORCE with PD-MIS and control variates
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0
    baseline = 0
    approximation = 0

    src_distributions = computePerDecisionMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config)

    # Keep track of useful statistics#
    stats = BatchStats(num_batch)

    if adaptive=="Yes":
        #Number of n_def next iteration
        n_def = computeInitialNdefPerDecisionMultipleImportanceWeights(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, ess_min)

    else:
        n_def = batch_size

    for i_batch in range(num_batch):

        stats.n_def[i_batch] = n_def
        batch_size = n_def

        [source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions ,n_def] = offPolicyUpdateMultipleImportanceSamplingCvPerDec(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, baseline, approximation, learning_rate, ess_min, adaptive, n_config_cv)

        # Update statistics
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        stats.policy_parameter[i_batch] = param
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess

    return stats

def offPolicyMultipleImportanceSamplingCvPdBaselineApproximated(env, batch_size, discount_factor, source_task, next_states_unclipped, clipped_actions, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, learning_rate, ess_min, adaptive, n_config_cv):
    """
    Perform transfer from source tasks, using REINFORCE with PD-MIS with baseline
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0
    baseline = 1
    approximation = 1

    src_distributions = computePerDecisionMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config)

    # Keep track of useful statistics#
    stats = BatchStats(num_batch)

    if adaptive=="Yes":
        #Number of n_def next iteration
        n_def = computeInitialNdefPerDecisionMultipleImportanceWeights(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, ess_min)

    else:
        n_def = batch_size

    for i_batch in range(num_batch):

        stats.n_def[i_batch] = n_def
        batch_size = n_def

        [source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions, n_def] = offPolicyUpdateMultipleImportanceSamplingCvPerDec(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, baseline, approximation, learning_rate, ess_min, adaptive, n_config_cv)

        # Update statistics
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        stats.policy_parameter[i_batch] = param
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess

    return stats

def offPolicyMultipleImportanceSamplingCvPdBaseline(env, batch_size, discount_factor, source_task, next_states_unclipped, clipped_actions, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch, learning_rate, ess_min, adaptive, n_config_cv):
    """
    Perform transfer from source tasks, using REINFORCE with PD-MIS with baseline
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param next_states_unclipped: matrix containing the unclipped next states of every episode for every time step
    :param clipped_actions: matrix containing the unclipped actions of every episode for every time step
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :param learning_rate: learning rate of the update rule
    :param ess_min: minimum effective sample size
    :param adaptive: yes if the algorithms has n_def adaptive, no if it hasn't
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0
    baseline = 1
    approximation = 0

    src_distributions = computePerDecisionMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config)

    # Keep track of useful statistics#
    stats = BatchStats(num_batch)

    if adaptive=="Yes":
        #Number of n_def next iteration
        n_def = computeInitialNdefPerDecisionMultipleImportanceWeights(param, env.A, source_param, variance_action, source_task, next_states_unclipped, clipped_actions, episodes_per_config, src_distributions, ess_min)

    else:
        n_def = batch_size

    for i_batch in range(num_batch):

        stats.n_def[i_batch] = n_def
        batch_size = n_def

        [source_param, source_task, next_states_unclipped, clipped_actions, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions, n_def] = offPolicyUpdateMultipleImportanceSamplingCvPerDecBaseline(env, param, source_param, episodes_per_config, source_task, next_states_unclipped, clipped_actions, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor, baseline, approximation, learning_rate, ess_min, adaptive, n_config_cv)

        # Update statistics
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        stats.policy_parameter[i_batch] = param
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess

    return stats
