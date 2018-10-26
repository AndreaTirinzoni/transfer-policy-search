import math as m
import numpy as np
import algorithmPolicySearch as alg
from collections import namedtuple

def optimalPolicy(env, num_episodes, discount_factor, batch_size, episode_length):
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
        stats.policy_parameter[i_batch] = K

        #print(state, action, reward, param)
    return stats

def createEpisode(env, episode_length, param, state, variance_action):
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

def computeGradientsSourceTarget(param, source_task, variance_action):
    """
        Compute the gradients estimation of the source targets with current policy

        Args:
            params: policy parameters
            source_task: source tasks
            variance_action: variance of the action's distribution

        Returns:
            A vector containing all the gradients
    """

    gradient_off_policy = np.sum(source_task[:, 2::3] - param * np.multiply(source_task[:, 1::3], source_task[:, 1::3]) / variance_action, axis=1)

    return gradient_off_policy

def computeImportanceWeightsSourceTarget(policy_param, env_param, source_param, variance_action, source_task):
    """
        Compute the importance weights considering policy and transition model_src

        Args:
            env: OpenAI environment
            param: current policy parameter
            source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
            variance_action: variance of the action's distribution
            source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
            episode_length: length of the episodes

        Returns:
            Returns the weights of the importance sampling
    """

    param_policy_src = source_param[:, 1] #policy parameter of source
    state_t = np.delete(source_task[:, 0::3], -1, axis=1)# state t
    state_t1 = source_task[:, 3::3] # state t+1
    action_t = source_task[:, 1::3] # action t
    variance_env = source_param[:, 4] # variance of the model transition
    A = source_param[:, 2] # environment parameter A of src
    B = source_param[:, 3] # environment parameter B of src
    policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-(action_t - policy_param*state_t)**2/(2*variance_action))
    policy_src = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-(action_t - np.multiply(param_policy_src, state_t.T).T)**2/(2*variance_action))
    model_tgt = np.multiply(1/np.sqrt(2*m.pi*variance_env), np.exp(np.divide(-(state_t1 - env_param * state_t - (B * action_t.T).T).T **2, (2*variance_env)).T).T).T
    model_src = np.multiply(1/np.sqrt(2*m.pi*variance_env), np.exp(np.divide(-(state_t1 - (A * state_t.T).T - (B * action_t.T).T).T **2, (2*variance_env)).T).T).T

    weights = np.prod(policy_tgt / policy_src * model_tgt / model_src, axis = 1)

    return weights

def offPolicyUpdate(env, param, source_param, episodes_per_config, source_task, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor):
    """
    Compute the gradient update of the policy parameter
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
    :return:
    Returns all informations related to the update, the new sorce parameter, source tasks, new parameter ...
    """
    #Compute gradients of the source task
    gradient_off_policy = computeGradientsSourceTarget(param, source_task, variance_action)
    #Compute importance weights_source_target of source task
    weights_source_target = computeImportanceWeightsSourceTarget(param, env.A, source_param, variance_action, source_task)
    # num_episodes_target = m.ceil((batch_size - 2*np.sum(weights_source_target) - m.sqrt(batch_size*(batch_size+4*(np.dot(weights_source_target, weights_source_target)-np.sum(weights_source_target)))))/2)
    num_episodes_target = batch_size
    episode_informations = np.zeros((num_episodes_target, 3))
    # Create new parameters and new tasks associated to episodes, used tu update the source_param and source_task later
    source_param_new = np.ones((num_episodes_target, 5))
    source_task_new = np.ones((num_episodes_target, episode_length*3+1))
    # Iterate for every episode in batch
    for i_episode in range(num_episodes_target):
        # Reset the environment and pick the first action
        state = env.reset()
        episode = createEpisode(env, episode_length, param, state, variance_action) # [state, action, reward, next_state]

        # Go through the episode and compute estimators

        total_return = np.sum(episode[:, 2])
        discounted_return = np.sum(np.multiply(np.power(discount_factor*np.ones(episode.shape[0]), range(episode.shape[0])), episode[:, 2]))
        gradient_est = np.sum((episode[:, 1] - param * np.multiply(episode[:, 0], episode[:, 0])) / variance_action)
        source_task_new[i_episode, 0::3] = np.concatenate((episode[:, 0].T, [episode[-1, 3]]))
        source_task_new[i_episode, 1::3] = episode[:, 1].T
        source_task_new[i_episode, 2::3] = episode[:, 2].T

        episode_informations[i_episode,:] = [gradient_est, total_return, discounted_return]
        source_param_new[i_episode, 0] = discounted_return
        source_param_new[i_episode, 1] = param
        source_param_new[i_episode, 2] = env.A
        source_param_new[i_episode, 3] = env.B
        source_param_new[i_episode, 4] = env.sigma_noise**2


    #Update the parameters
    N = weights_source_target.shape[0] + num_episodes_target
    weights_source_target_update = np.concatenate([weights_source_target, np.ones(num_episodes_target)], axis=0) # are the weights used for computing ESS
    gradient_off_policy_update = np.concatenate([gradient_off_policy, episode_informations[:,0]], axis=0)
    discounted_rewards_all = np.concatenate([source_task[:,1], episode_informations[:,2]], axis=0)
    gradient = 1/N * np.dot(np.multiply(weights_source_target_update, gradient_off_policy_update), discounted_rewards_all)
    param, t, m_t, v_t = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)

    #Compute rewards of batch
    tot_reward_batch = np.mean(episode_informations[:,1])
    discounted_reward_batch = np.mean(episode_informations[:,2])

    # Concatenate new episodes to source tasks
    source_param = np.concatenate((source_param, source_param_new), axis=0)
    source_task = np.concatenate((source_task, source_task_new), axis=0)
    episodes_per_config = np.concatenate((episodes_per_config, [batch_size]))
    return source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch

def offPolicyImportanceSampling(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch):
    """
        Perform transfer from source tasks, using REINFORCE with IS

        Args:
            env: OpenAI environment
            batch_size: size of the batch
            discount_factor: the discout factor
            source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
            source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
            episodes_per_config: number of episodes for every policy_parameter - env_parameter
            variance_action: variance of the action's distribution
            episode_length: length of the episodes
            mean_initial_param: mean initial policy parameter
            num_batch: number of batch for REINFORCE

        Returns:
            An EpisodeStats object with two numpy arrays for episode_disc_reward and episode_rewards.

    """
    param = initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Keeps track of useful statistics#
    stats = EpisodeStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch),
        policy_parameter=np.zeros(num_batch))

    for i_batch in range(num_batch):
        stats.policy_parameter[i_batch] = param
        [source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch] = offPolicyUpdate(env, param, source_param, episodes_per_config, source_task, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor)
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch
    return stats

EpisodeStats = namedtuple("Stats",["episode_total_rewards", "episode_disc_rewards", "policy_parameter"])
