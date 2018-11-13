import math as m
import numpy as np
from collections import namedtuple
import algorithmPolicySearch as alg

BatchStats = namedtuple("Stats",["episode_total_rewards", "episode_disc_rewards", "policy_parameter", "gradient", "ess"])

def optimalPolicy(env, num_episodes, batch_size, discount_factor, variance_action, episode_length):
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
        policy_parameter=np.zeros(num_batch),
        gradient=np.zeros(num_batch),
        ess=np.zeros(num_batch))
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

def createBatch(env, batch_size, episode_length, param, variance_action):
    """
    Create a batch of episodes
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param episode_length: length of the episode
    :param param: policy parameter
    :param variance_action: variance of the action's distribution
    :return: A tensor containing [num episodes, timestep, informations] where informations stays for: [state, action, reward, next_state]
    """

    batch = np.zeros((batch_size, episode_length, 4)) # [state, action, reward, next_state]
    for i_batch in range(batch_size):
        state = env.reset()

        for t in range(episode_length):
            #env.render()
            # Take a step
            mean_action = param*state
            action = np.random.normal(mean_action, m.sqrt(variance_action))
            next_state, reward, done, _ = env.step(action)
            # Keep track of the transition

            #print(state, action, reward, param)
            batch[i_batch, t, :] = [state, action, reward, next_state]

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

    gradient_off_policy = (action_t - param * state_t) * state_t / variance_action

    return gradient_off_policy

#Compute weights of different estimators

def computeImportanceWeightsSourceTarget(policy_param, env_param, source_param, variance_action, source_task):
    """
    Compute the importance weights considering policy and transition model
    :param policy_param: current policy parameter
    :param env_param: current environment parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :return: Returns the weights of the importance sampling estimator
    """

    param_policy_src = source_param[:, 1][:, np.newaxis] #policy parameter of source
    state_t = np.delete(source_task[:, 0::3], -1, axis=1)# state t
    state_t1 = source_task[:, 3::3] # state t+1
    action_t = source_task[:, 1::3] # action t
    variance_env = source_param[:, 4][:, np.newaxis] # variance of the model transition
    A = source_param[:, 2][:, np.newaxis] # environment parameter A of src
    B = source_param[:, 3][:, np.newaxis] # environment parameter B of src

    policy_tgt = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((action_t - policy_param * state_t)**2)/(2*variance_action)), axis=1)
    policy_src = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((action_t - param_policy_src * state_t)**2)/(2*variance_action)), axis=1)

    model_tgt = np.prod(1/np.sqrt(2*m.pi*variance_env) * np.exp(-((state_t1 - env_param * state_t - B * action_t) **2) / (2*variance_env)), axis=1)
    model_src = np.prod(1/np.sqrt(2*m.pi*variance_env) * np.exp(-((state_t1 - A * state_t - B * action_t) **2) / (2*variance_env)), axis=1)

    weights = policy_tgt / policy_src * model_tgt / model_src

    return weights

def computeImportanceWeightsSourceTargetPerDecision(policy_param, env_param, source_param, variance_action, source_task):
    """
    Compute the per-decision importance weights considering policy and transition model
    :param policy_param: current policy parameter
    :param env_param: current environment parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :return: Returns the weights of the per-decision importance sampling estimator
    """

    param_policy_src = source_param[:, 1][:, np.newaxis] #policy parameter of source
    state_t = np.delete(source_task[:, 0::3], -1, axis=1)# state t
    state_t1 = source_task[:, 3::3] # state t+1
    action_t = source_task[:, 1::3] # action t
    variance_env = source_param[:, 4][:, np.newaxis] # variance of the model transition
    A = source_param[:, 2][:, np.newaxis] # environment parameter A of src
    B = source_param[:, 3][:, np.newaxis] # environment parameter B of src

    policy_tgt = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((action_t - policy_param * state_t)**2)/(2*variance_action)), axis=1)
    policy_src = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((action_t - param_policy_src * state_t)**2)/(2*variance_action)), axis=1)

    model_tgt = np.cumprod(1/np.sqrt(2*m.pi*variance_env) * np.exp(-((state_t1 - env_param * state_t - B * action_t) **2) / (2*variance_env)), axis=1)
    model_src = np.cumprod(1/np.sqrt(2*m.pi*variance_env) * np.exp(-((state_t1 - A * state_t - B * action_t) **2) / (2*variance_env)), axis=1)

    weights = policy_tgt / policy_src * model_tgt / model_src

    return weights

def computeMultipleImportanceWeightsSourceTarget(policy_param, env_param, source_param, variance_action, source_task, episodes_per_config, src_distributions):
    """
    Compute the multiple importance weights considering policy and transition model
    :param policy_param: current policy parameter
    :param env_param: current environment parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
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
    state_t1 = np.repeat(source_task[:, 3::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t+1
    action_t = np.repeat(source_task[:, 1::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # action t
    variance_env = source_param[:, 4][:, np.newaxis, np.newaxis] # variance of the model transition

    policy_src_new_traj = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((action_t[evaluated_trajectories:, :, :] - param_policy_src * state_t[evaluated_trajectories:, :, :])**2)/(2*variance_action)), axis=1)
    model_src_new_traj = np.prod(1/np.sqrt(2*m.pi*variance_env[evaluated_trajectories:, :, :]) * np.exp(-((state_t1[evaluated_trajectories:, :, :] - A * state_t[evaluated_trajectories:, :, :] - B * action_t[evaluated_trajectories:, :, :]) **2) / (2*variance_env[evaluated_trajectories:, :, :])), axis=1)

    policy_src_new_param = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((action_t[0:evaluated_trajectories, :, 0] - param_policy_src[0:evaluated_trajectories, :, -1] * state_t[0:evaluated_trajectories, :, 0])**2)/(2*variance_action)), axis=1)
    model_src_new_param = np.prod(1/np.sqrt(2*m.pi*variance_env[0:evaluated_trajectories, :, 0]) * np.exp(-((state_t1[0:evaluated_trajectories, :, 0] - A[0:evaluated_trajectories, :, -1] * state_t[0:evaluated_trajectories, :, 0] - B[0:evaluated_trajectories, :, -1] * action_t[0:evaluated_trajectories, :, 0]) **2) / (2*variance_env[0:evaluated_trajectories, :, 0])), axis=1)

    src_distributions = np.concatenate((src_distributions, np.matrix(policy_src_new_param * model_src_new_param).T), axis=1)
    src_distributions = np.concatenate((src_distributions, np.matrix(policy_src_new_traj * model_src_new_traj)), axis=0)

    policy_tgt = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((action_t[:, :, 0] - policy_param*state_t[:, :, 0])**2)/(2*variance_action)), axis = 1)
    model_tgt = np.prod(1/np.sqrt(2*m.pi*variance_env[:, :, 0]) * np.exp(-((state_t1[:, :, 0] - env_param * state_t[:, :, 0] - B[:, :, -1] * action_t[:, :, 0]) **2) / (2*variance_env[:, :, 0])), axis = 1)

    mis_denominator = np.dot(episodes_per_config/n, src_distributions.T).T

    weights = policy_tgt[:, np.newaxis] * model_tgt[:, np.newaxis] / mis_denominator

    return weights, src_distributions

def computeMultipleImportanceWeightsSourceTargetPerDecision(policy_param, env_param, source_param, variance_action, source_task, episodes_per_config, src_distributions):
    """
    Compute the per-decision multiple importance weights considering policy and transition model
    :param policy_param: current policy parameter
    :param env_param: current environment parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
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
    state_t1 = np.repeat(source_task[:, 3::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t+1
    action_t = np.repeat(source_task[:, 1::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # action t
    variance_env = source_param[:, 4][:, np.newaxis, np.newaxis] # variance of the model transition

    policy_src_new_traj = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((action_t[evaluated_trajectories:, :, :] - param_policy_src * state_t[evaluated_trajectories:, :, :])**2)/(2*variance_action)), axis=1)
    model_src_new_traj = np.cumprod(1/np.sqrt(2*m.pi*variance_env[evaluated_trajectories:, :, :]) * np.exp(-((state_t1[evaluated_trajectories:, :, :] - A * state_t[evaluated_trajectories:, :, :] - B * action_t[evaluated_trajectories:, :, :]) **2) / (2*variance_env[evaluated_trajectories:, :, :])), axis=1)

    policy_src_new_param = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((action_t[0:evaluated_trajectories, :, 0] - param_policy_src[0:evaluated_trajectories, :, -1] * state_t[0:evaluated_trajectories, :, 0])**2)/(2*variance_action)), axis=1)
    model_src_new_param = np.cumprod(1/np.sqrt(2*m.pi*variance_env[0:evaluated_trajectories, :, 0]) * np.exp(-((state_t1[0:evaluated_trajectories, :, 0] - A[0:evaluated_trajectories, :, -1] * state_t[0:evaluated_trajectories, :, 0] - B[0:evaluated_trajectories, :, -1] * action_t[0:evaluated_trajectories, :, 0]) **2) / (2*variance_env[0:evaluated_trajectories, :, 0])), axis=1)

    src_distributions = np.concatenate((src_distributions, (policy_src_new_param * model_src_new_param)[:, :, np.newaxis]), axis=2)
    src_distributions = np.concatenate((src_distributions, (policy_src_new_traj * model_src_new_traj)), axis=0)

    policy_tgt = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((action_t[:, :, 0] - policy_param*state_t[:, :, 0])**2)/(2*variance_action)), axis = 1)
    model_tgt = np.cumprod(1/np.sqrt(2*m.pi*variance_env[:, :, 0]) * np.exp(-((state_t1[:, :, 0] - env_param * state_t[:, :, 0] - (B[:, :, -1] * action_t)[:, :, 0]) **2) / (2*variance_env[:, :, 0])), axis = 1)

    mis_denominator = np.sum(episodes_per_config/n * src_distributions, axis=2)

    weights = policy_tgt * model_tgt / mis_denominator

    return weights, src_distributions

# Compute the update of the algorithms using different estimators

def offPolicyUpdateImportanceSamplingPerDec(env, param, source_param, episodes_per_config, source_task, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor):
    """
    Compute the gradient update of the policy parameter using per-decision importance sampling
    :param env: OpenAI environment
    :param param: current policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param batch_size: size of the batch
    :param t: parameter of ADAM
    :param m_t: parameter of ADAM
    :param v_t: parameter of ADAM
    :param discount_factor: the discout factor
    :return: Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """

    #Compute gradients of the source task
    gradient_off_policy = np.cumsum(computeGradientsSourceTargetTimestep(param, source_task, variance_action), axis=1)
    #Compute importance weights_source_target of source task
    weights_source_target = computeImportanceWeightsSourceTargetPerDecision(param, env.A, source_param, variance_action, source_task)
    num_episodes_target = batch_size
    # Create new parameters and new tasks associated to episodes, used tu update the source_param and source_task later
    source_param_new = np.ones((num_episodes_target, 5))
    source_task_new = np.ones((num_episodes_target, episode_length*3+1))
    # Iterate for every episode in batch

    batch = createBatch(env, batch_size, episode_length, param, variance_action) # [state, action, reward, next_state]

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))
    # The return after this timestep
    total_return = np.sum(batch[:, :, 2], axis=1)
    discounted_return_timestep = discount_factor_timestep * batch[:, :, 2]
    discounted_return = np.sum(discounted_return_timestep, axis=1)
    gradient_est_timestep = np.cumsum(((batch[:, :, 1] - param * batch[:, :, 0]) * batch[:, :, 0]) / variance_action, axis=1)

    source_param_new[:, 0] = discounted_return
    source_param_new[:, 1] = param
    source_param_new[:, 2] = env.A
    source_param_new[:, 3] = env.B
    source_param_new[:, 4] = env.sigma_noise**2

    source_task_new[:, 0::3] = np.concatenate((batch[:, :, 0], np.matrix(batch[:, -1, 3]).T), axis=1)
    source_task_new[:, 1::3] = batch[:, :, 1]
    source_task_new[:, 2::3] = batch[:, :, 2]

    #Update the parameters
    N =  source_task.shape[0] + batch_size
    weights_source_target_update = np.concatenate([weights_source_target, np.ones((num_episodes_target, episode_length))], axis=0) # are the weights used for computing ESS
    weights_source_target_update[np.isnan(weights_source_target_update)] = 0
    gradient_off_policy_update = np.concatenate([gradient_off_policy, gradient_est_timestep], axis=0)
    discounted_rewards_all = np.concatenate([discount_factor_timestep * source_task[:,2::3], discounted_return_timestep], axis=0)
    gradient = 1/N * np.sum(np.sum(weights_source_target_update * gradient_off_policy_update * discounted_rewards_all, axis = 1))
    param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    #param = param + 0.01 * gradient
    #Compute rewards of batch
    tot_reward_batch = np.mean(total_return)
    discounted_reward_batch = np.mean(discounted_return)
    ess = np.min(np.linalg.norm(weights_source_target_update, 1, axis=0)**2 / np.linalg.norm(weights_source_target_update, 2, axis=0)**2, axis=0)
    # Concatenate new episodes to source tasks
    source_param = np.concatenate((source_param, source_param_new), axis=0)
    source_task = np.concatenate((source_task, source_task_new), axis=0)
    episodes_per_config = np.concatenate((episodes_per_config, [batch_size]))
    return source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess

def offPolicyUpdateImportanceSampling(env, param, source_param, episodes_per_config, source_task, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor):
    """
    Compute the gradient update of the policy parameter using importance sampling
    :param env: OpenAI environment
    :param param: current policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param batch_size: size of the batch
    :param t: parameter of ADAM
    :param m_t: parameter of ADAM
    :param v_t: parameter of ADAM
    :param discount_factor: the discount factor
    :return: Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """
    #Compute gradients of the source task
    gradient_off_policy = np.sum(computeGradientsSourceTargetTimestep(param, source_task, variance_action), axis=1)
    #Compute importance weights_source_target of source task
    weights_source_target = computeImportanceWeightsSourceTarget(param, env.A, source_param, variance_action, source_task)
    # num_episodes_target = m.ceil((batch_size - 2*np.sum(weights_source_target) - m.sqrt(batch_size*(batch_size+4*(np.dot(weights_source_target, weights_source_target)-np.sum(weights_source_target)))))/2)
    num_episodes_target = batch_size
    # Create new parameters and new tasks associated to episodes, used tu update the source_param and source_task later
    source_param_new = np.ones((num_episodes_target, 5))
    source_task_new = np.ones((num_episodes_target, episode_length*3+1))
    # Iterate for every episode in batch

    batch = createBatch(env, batch_size, episode_length, param, variance_action) # [state, action, reward, next_state]

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))
    # The return after this timestep
    total_return = np.sum(batch[:, :, 2], axis=1)
    discounted_return = np.sum((discount_factor_timestep * batch[:, :, 2]), axis=1)
    gradient_est = np.sum(((batch[:, :, 1] - param * batch[:, :, 0]) * batch[:, :, 0]) / variance_action, axis=1)

    source_param_new[:, 0] = discounted_return
    source_param_new[:, 1] = param
    source_param_new[:, 2] = env.A
    source_param_new[:, 3] = env.B
    source_param_new[:, 4] = env.sigma_noise**2

    source_task_new[:, 0::3] = np.concatenate((batch[:, :, 0], np.matrix(batch[:, -1, 3]).T), axis=1)
    source_task_new[:, 1::3] = batch[:, :, 1]
    source_task_new[:, 2::3] = batch[:, :, 2]

    #Update the parameters
    N = source_task.shape[0] + num_episodes_target
    weights_source_target_update = np.concatenate([weights_source_target, np.ones(num_episodes_target)], axis=0) # are the weights used for computing ESS
    weights_source_target_update[np.isnan(weights_source_target_update)] = 0
    gradient_off_policy_update = np.concatenate([gradient_off_policy, np.squeeze(np.asarray(gradient_est))], axis=0)
    discounted_rewards_all = np.concatenate([source_param[:,0], np.squeeze(np.asarray(discounted_return))], axis=0)
    gradient = 1/N * np.sum((weights_source_target_update * gradient_off_policy_update) * discounted_rewards_all, axis=0)
    param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    #param = param + 0.01 * gradient
    #Compute rewards of batch
    tot_reward_batch = np.mean(total_return)
    discounted_reward_batch = np.mean(discounted_return)
    ess = np.linalg.norm(weights_source_target_update, 1)**2 / np.linalg.norm(weights_source_target_update, 2)**2

    # Concatenate new episodes to source tasks
    source_param = np.concatenate((source_param, source_param_new), axis=0)
    source_task = np.concatenate((source_task, source_task_new), axis=0)
    episodes_per_config = np.concatenate((episodes_per_config, [batch_size]))
    return source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess

def offPolicyUpdateMultipleImportanceSampling(env, param, source_param, episodes_per_config, source_task, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor):
    """
    Compute the gradient update of the policy parameter using multiple importance sampling
    :param env: OpenAI environment
    :param param: current policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param variance_action: variance of the action's distribution
    :param src_distributions: source distributions, (the qj of the MIS denominator policy)
    :param episode_length: length of the episodes
    :param batch_size: size of the batch
    :param t: parameter of ADAM
    :param m_t: parameter of ADAM
    :param v_t: parameter of ADAM
    :param discount_factor: the discout factor
    :return: Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """

    #Compute importance weights_source_target of source task
    # num_episodes_target = m.ceil((batch_size - 2*np.sum(weights_source_target) - m.sqrt(batch_size*(batch_size+4*(np.dot(weights_source_target, weights_source_target)-np.sum(weights_source_target)))))/2)
    num_episodes_target = batch_size
    # Create new parameters and new tasks associated to episodes, used tu update the source_param and source_task later
    source_param_new = np.ones((num_episodes_target, 5))
    source_task_new = np.ones((num_episodes_target, episode_length*3+1))
    # Iterate for every episode in batch

    batch = createBatch(env, batch_size, episode_length, param, variance_action) # [state, action, reward, next_state]

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))
    # The return after this timestep
    total_return = np.sum(batch[:, :, 2], axis=1)
    discounted_return = np.sum((discount_factor_timestep * batch[:, :, 2]), axis=1)

    source_param_new[:, 0] = discounted_return
    source_param_new[:, 1] = param
    source_param_new[:, 2] = env.A
    source_param_new[:, 3] = env.B
    source_param_new[:, 4] = env.sigma_noise**2

    source_task_new[:, 0::3] = np.concatenate((batch[:, :, 0], np.matrix(batch[:, -1, 3]).T), axis=1)
    source_task_new[:, 1::3] = batch[:, :, 1]
    source_task_new[:, 2::3] = batch[:, :, 2]

    # Concatenate new episodes to source tasks
    source_param = np.concatenate((source_param, source_param_new), axis=0)
    source_task = np.concatenate((source_task, source_task_new), axis=0)
    episodes_per_config = np.concatenate((episodes_per_config, [batch_size]))

    #Update the parameters
    N =  source_task.shape[0]

    #Compute gradients of the source task
    [weights_source_target_update, src_distributions] = computeMultipleImportanceWeightsSourceTarget(param, env.A, source_param, variance_action, source_task, episodes_per_config, src_distributions)
    weights_source_target_update[np.isnan(weights_source_target_update)] = 0
    gradient_off_policy_update = np.sum(computeGradientsSourceTargetTimestep(param, source_task, variance_action), axis=1)
    discounted_rewards_all = source_param[:,0]
    gradient = 1/N * np.sum((np.squeeze(np.array(weights_source_target_update)) * gradient_off_policy_update) * discounted_rewards_all)
    param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    #param = param + 0.01 * gradient
    #Compute rewards of batch
    tot_reward_batch = np.mean(total_return)
    discounted_reward_batch = np.mean(discounted_return)
    ess = np.linalg.norm(weights_source_target_update, 1)**2 / np.linalg.norm(weights_source_target_update, 2)**2

    return source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions

def offPolicyUpdateMultipleImportanceSamplingPerDec(env, param, source_param, episodes_per_config, source_task, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor):
    """
    Compute the gradient update of the policy parameter using per decision multiple importance sampling
    :param env: OpenAI environment
    :param param: current policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param variance_action: variance of the action's distribution
    :param src_distributions: source distributions, (the qj of the PD-MIS denominator policy)
    :param episode_length: length of the episodes
    :param batch_size: size of the batch
    :param t: parameter of ADAM
    :param m_t: parameter of ADAM
    :param v_t: parameter of ADAM
    :param discount_factor: the discout factor
    :return: Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """

    #Compute gradients of the source task
    num_episodes_target = batch_size
    # Create new parameters and new tasks associated to episodes, used tu update the source_param and source_task later
    source_param_new = np.ones((num_episodes_target, 5))
    source_task_new = np.ones((num_episodes_target, episode_length*3+1))
    # Iterate for every episode in batch

    batch = createBatch(env, batch_size, episode_length, param, variance_action) # [state, action, reward, next_state]

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))
    # The return after this timestep
    total_return = np.sum(batch[:, :, 2], axis=1)
    discounted_return_timestep = (discount_factor_timestep * batch[:, :, 2])

    source_param_new[:, 0] = np.sum(discounted_return_timestep, axis=1)
    source_param_new[:, 1] = param
    source_param_new[:, 2] = env.A
    source_param_new[:, 3] = env.B
    source_param_new[:, 4] = env.sigma_noise**2

    source_task_new[:, 0::3] = np.concatenate((batch[:, :, 0], np.matrix(batch[:, -1, 3]).T), axis=1)
    source_task_new[:, 1::3] = batch[:, :, 1]
    source_task_new[:, 2::3] = batch[:, :, 2]

    # Concatenate new episodes to source tasks
    source_param = np.concatenate((source_param, source_param_new), axis=0)
    source_task = np.concatenate((source_task, source_task_new), axis=0)
    episodes_per_config = np.concatenate((episodes_per_config, [batch_size]))

    #Update the parameters
    N =  source_task.shape[0]
    #Compute importance weights_source_target of source task
    [weights_source_target_update, src_distributions] = computeMultipleImportanceWeightsSourceTargetPerDecision(param, env.A, source_param, variance_action, source_task, episodes_per_config, src_distributions)
    weights_source_target_update[np.isnan(weights_source_target_update)] = 0
    gradient_off_policy_update = np.cumsum(computeGradientsSourceTargetTimestep(param, source_task, variance_action), axis=1)
    discounted_rewards_all = discount_factor_timestep * source_task[:,2::3]
    gradient = 1/N * np.sum(np.sum((weights_source_target_update * gradient_off_policy_update) * discounted_rewards_all, axis = 1), axis=0)
    param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    #param = param + 0.01 * gradient
    #Compute rewards of batch
    tot_reward_batch = np.mean(total_return)
    discounted_reward_batch = np.mean(np.sum(discounted_return_timestep, axis=1))
    ess = np.min(np.linalg.norm(weights_source_target_update, 1, axis=0)**2 / np.linalg.norm(weights_source_target_update, 2, axis=0)**2, axis=0)

    return source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions

def offPolicyUpdateMultipleImportanceSamplingControlVariates(env, param, source_param, episodes_per_config, source_task, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor):
    """
    Compute the gradient update of the policy parameter using multiple importance sampling
    :param env: OpenAI environment
    :param param: current policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: number of episodes for every policy_parameter - env_parameter
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param batch_size: size of the batch
    :param t: parameter of ADAM
    :param m_t: parameter of ADAM
    :param v_t: parameter of ADAM
    :param discount_factor: the discout factor
    :return: Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """

    #Compute importance weights_source_target of source task
    # num_episodes_target = m.ceil((batch_size - 2*np.sum(weights_source_target) - m.sqrt(batch_size*(batch_size+4*(np.dot(weights_source_target, weights_source_target)-np.sum(weights_source_target)))))/2)
    num_episodes_target = batch_size
    # Create new parameters and new tasks associated to episodes, used tu update the source_param and source_task later
    source_param_new = np.ones((num_episodes_target, 5))
    source_task_new = np.ones((num_episodes_target, episode_length*3+1))
    # Iterate for every episode in batch

    batch = createBatch(env, batch_size, episode_length, param, variance_action) # [state, action, reward, next_state]

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))
    # The return after this timestep
    total_return = np.sum(batch[:, :, 2], axis=1)
    discounted_return = np.sum((discount_factor_timestep * batch[:, :, 2]), axis=1)

    source_param_new[:, 0] = discounted_return
    source_param_new[:, 1] = param
    source_param_new[:, 2] = env.A
    source_param_new[:, 3] = env.B
    source_param_new[:, 4] = env.sigma_noise**2

    # Concatenate new episodes to source tasks
    source_param = np.concatenate((source_param, source_param_new), axis=0)
    source_task = np.concatenate((source_task, source_task_new), axis=0)
    episodes_per_config = np.concatenate((episodes_per_config, [batch_size]))

    #Update the parameters
    N =  source_task.shape[0] + num_episodes_target

    #Compute gradients of the source task
    weights_source_target_update = computeMultipleImportanceWeightsSourceTarget(param, env.A, source_param, variance_action, source_task, episodes_per_config, batch_size, episode_length)
    gradient_off_policy_update = np.sum(computeGradientsSourceTargetTimestep(param, source_task, variance_action), axis=1)
    discounted_rewards_all = source_param[:,0]
    gradient = 1/N * np.sum((weights_source_target_update * gradient_off_policy_update) * discounted_rewards_all, axis=0)
    param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    #param = param + 0.01 * gradient
    #Compute rewards of batch
    tot_reward_batch = np.mean(total_return)
    discounted_reward_batch = np.mean(discounted_return)

    return source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch

# Algorithm off policy using different estimators

def offPolicyImportanceSampling(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch):
    """
    Perform transfer from source tasks, using REINFORCE with IS
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Keeps track of useful statistics#
    stats = BatchStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch),
        policy_parameter=np.zeros(num_batch),
        gradient=np.zeros(num_batch),
        ess=np.zeros(num_batch))

    for i_batch in range(num_batch):
        stats.policy_parameter[i_batch] = param
        [source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess] = offPolicyUpdateImportanceSampling(env, param, source_param, episodes_per_config, source_task, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor)
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess
    return stats

def offPolicyImportanceSamplingPd(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch):
    """
    Perform transfer from source tasks, using REINFORCE with PD-IS
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Keeps track of useful statistics#
    stats = BatchStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch),
        policy_parameter=np.zeros(num_batch),
        gradient=np.zeros(num_batch),
        ess=np.zeros(num_batch))

    for i_batch in range(num_batch):
        stats.policy_parameter[i_batch] = param
        [source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess] = offPolicyUpdateImportanceSamplingPerDec(env, param, source_param, episodes_per_config, source_task, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor)
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess
    return stats

def computeMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, episodes_per_config):
    """
    Compute the qj of the MIS denominator
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action:
    :param source_task:
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :return: a matrix (N trajectories * M parameters) containing the qj related to the denominator of the MIS weights
    """

    param_indices = np.concatenate(([0], np.cumsum(np.delete(episodes_per_config, -1))))

    param_policy_src = source_param[param_indices, 1][np.newaxis, np.newaxis, :]#policy parameter of source not repeated
    variance_env = source_param[:, 4][:, np.newaxis, np.newaxis] # variance of the model transition
    A = source_param[param_indices, 2][np.newaxis, np.newaxis, :] # environment parameter A of src
    B = source_param[param_indices, 3][np.newaxis, np.newaxis, :] # environment parameter B of src

    state_t = np.repeat(np.delete(source_task[:, 0::3], -1, axis=1)[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t
    state_t1 = np.repeat(source_task[:, 3::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t+1
    action_t = np.repeat(source_task[:, 1::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # action t

    src_distributions_policy = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((action_t - param_policy_src * state_t)**2)/(2*variance_action)), axis=1)

    src_distributions_model = np.prod(1/np.sqrt(2*m.pi*variance_env) * np.exp(-((state_t1 - A * state_t - B * action_t) **2) / (2*variance_env)), axis=1)

    return src_distributions_model * src_distributions_policy

def offPolicyMultipleImportanceSampling(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch):
    """
    Perform transfer from source tasks, using REINFORCE with MIS
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    print("MIS")
    src_distributions = computeMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, episodes_per_config)
    # Keeps track of useful statistics#
    stats = BatchStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch),
        policy_parameter=np.zeros(num_batch),
        gradient=np.zeros(num_batch),
        ess=np.zeros(num_batch))

    for i_batch in range(num_batch):
        print("Batch: " + str(i_batch))
        stats.policy_parameter[i_batch] = param
        [source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions] = offPolicyUpdateMultipleImportanceSampling(env, param, source_param, episodes_per_config, source_task, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor)
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess
    return stats

def computePerDecisionMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, episodes_per_config):
    """
    Compute the qj of the PD-MIS denominator
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param variance_action: variance of the action's distribution
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :return: a tensor (N trajectories * T timesteps * M parameters) containing the qj related to the denominator of the PD-MIS weights
    """

    param_indices = np.concatenate(([0], np.cumsum(np.delete(episodes_per_config, -1))))

    param_policy_src = source_param[param_indices, 1][np.newaxis, np.newaxis, :]#policy parameter of source not repeated
    variance_env = source_param[:, 4][:, np.newaxis, np.newaxis]# variance of the model transition
    A = source_param[param_indices, 2][np.newaxis, np.newaxis, :] # environment parameter A of src
    B = source_param[param_indices, 3][np.newaxis, np.newaxis, :] # environment parameter B of src

    state_t = np.repeat(np.delete(source_task[:, 0::3], -1, axis=1)[:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t
    state_t1 = np.repeat(source_task[:, 3::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # state t+1
    action_t = np.repeat(source_task[:, 1::3][:, :, np.newaxis], param_policy_src.shape[2], axis=2) # action t

    src_distributions_policy = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((action_t - param_policy_src * state_t)**2)/(2*variance_action)), axis=1)

    src_distributions_model = np.cumprod(1/np.sqrt(2*m.pi*variance_env) * np.exp(-((state_t1 - A * state_t - B * action_t) **2) / (2*variance_env)), axis=1)

    return src_distributions_model * src_distributions_policy

def offPolicyMultipleImportanceSamplingPd(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch):
    """
    Perform transfer from source tasks, using REINFORCE with PD-MIS
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0
    print("PD-MIS")
    src_distributions = computePerDecisionMultipleImportanceWeightsSourceDistributions(source_param, variance_action, source_task, episodes_per_config)
    # Keeps track of useful statistics#
    stats = BatchStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch),
        policy_parameter=np.zeros(num_batch),
        gradient=np.zeros(num_batch),
        ess=np.zeros(num_batch))

    for i_batch in range(num_batch):
        print("Batch: " + str(i_batch))
        stats.policy_parameter[i_batch] = param
        [source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, src_distributions] = offPolicyUpdateMultipleImportanceSamplingPerDec(env, param, source_param, episodes_per_config, source_task, src_distributions, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor)
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch
        stats.gradient[i_batch] = gradient
        stats.ess[i_batch] = ess
    return stats

def offPolicyMultipleImportanceSamplingControlVariates(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch):
    """
    Perform transfer from source tasks, using REINFORCE with PD-MIS
    :param env: OpenAI environment
    :param batch_size: size of the batch
    :param discount_factor: the discout factor
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param episodes_per_config: vector containing the number of episodes for every policy_parameter - env_parameter configuration
    :param variance_action: variance of the action's distribution
    :param episode_length: length of the episodes
    :param initial_param: initial policy parameter
    :param num_batch: number of batch of the algorithm
    :return: Return a BatchStats object
    """

    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Keeps track of useful statistics#
    stats = BatchStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch),
        policy_parameter=np.zeros(num_batch))

    for i_batch in range(num_batch):
        stats.policy_parameter[i_batch] = param
        [source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch] = offPolicyUpdateMultipleImportanceSamplingControlVariates(env, param, source_param, episodes_per_config, source_task, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor)
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch
    return stats