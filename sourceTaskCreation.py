import gym
import envs
import numpy as np
import math as m

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

    batch = np.zeros((batch_size, episode_length, 6)) # [state, action, reward, next_state]
    for i_batch in range(batch_size):
        state = env.reset()

        for t in range(episode_length):
            # Take a step
            mean_action = param*state
            action = np.random.normal(mean_action, m.sqrt(variance_action))
            next_state, reward, done, unclipped_state, clipped_action = env.step(action)
            # Keep track of the transition

            batch[i_batch, t, :] = [state, clipped_action, reward, next_state, unclipped_state, action]

            if done:
                break

            state = next_state

    return batch

def sourceTaskCreation(episode_length, batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max):
    """
    Creates a source dataset
    :param env: OpenAI environment
    :param episode_length: length of each episode
    :param batch_size: size of every batch
    :param discount_factor: the discount factor
    :param variance_action: the variance of the action's distribution
    :param env_param_min: the minimum value of the environment's parameter
    :param env_param_max: the maximum value of the environment's parameter
    :param policy_param_min: the minimum value of the policy's parameter
    :param policy_param_max: the maximum value of the policy's parameter
    :return:A data structure containing all informations about the episodes,
            a data structure containing informations about the parameters of
            the episodes and a vector containing the number of episodes for every configuration
    """

    policy_param = np.linspace(policy_param_min, policy_param_max, 20)
    env_param = np.linspace(env_param_min, env_param_max, 40)
    i_episode = 0
    episodes_per_configuration = np.zeros(policy_param.shape[0]*env_param.shape[0])
    i_configuration = 0
    episode_per_param = batch_size
    length_source_task = policy_param.shape[0]*env_param.shape[0]*episode_per_param
    source_task = np.zeros((length_source_task, episode_length*3+1)) # every line a task, every task has all [clipped_state, action, reward]
    # Every line is a task, every task has [discounted_return, policy_parameter, A, B, variance, parameter_configuration id] where A and B are the evironment params
    source_param = np.zeros((length_source_task, 5))
    next_states_unclipped = np.zeros((length_source_task, episode_length))
    actions_clipped = np.zeros((length_source_task, episode_length))

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    for i_policy_param in range(policy_param.shape[0]):

        for i_env_param in range(env_param.shape[0]):

            env.setA(env_param[i_env_param])

            # Reset the environment and pick the first action
            batch = createBatch(env, episode_per_param, episode_length, policy_param[i_policy_param], variance_action) # [state, action, reward, next_state]

            #  Go through the episode and compute estimators

            discounted_return = np.sum((discount_factor_timestep * batch[:, :, 2]), axis=1)

            #I populate the source task
            source_task[i_episode:i_episode+episode_per_param, 0::3] = np.concatenate((batch[:, :, 0], np.matrix(batch[:, -1, 3]).T), axis=1)
            source_task[i_episode:i_episode+episode_per_param, 1::3] = batch[:, :, 5]
            source_task[i_episode:i_episode+episode_per_param, 2::3] = batch[:, :, 2]

            #unclipped next_states and actions
            next_states_unclipped[i_episode:i_episode+episode_per_param, :] = batch[:, :, 4]
            actions_clipped[i_episode:i_episode+episode_per_param, :] = batch[:, :, 1]

            #I populate the source parameters
            source_param[i_episode:i_episode+episode_per_param, 0] = discounted_return
            source_param[i_episode:i_episode+episode_per_param, 1] = policy_param[i_policy_param]
            source_param[i_episode:i_episode+episode_per_param, 2] = env.A
            source_param[i_episode:i_episode+episode_per_param, 3] = env.B
            source_param[i_episode:i_episode+episode_per_param, 4] = env.sigma_noise**2

            i_episode += episode_per_param

            episodes_per_configuration[i_configuration] = episode_per_param
            i_configuration += 1


    return source_task, source_param, episodes_per_configuration.astype(int), next_states_unclipped, actions_clipped

env = gym.make('LQG1D-v0')
# variance_action = 0.1
# episode_length = 20
# np.random.seed(2000)
# num_episodes=1000
# batch_size = 10
# discount_factor = 0.99
# env_param_min = 0.5
# env_param_max = 1.5
# policy_param_min = -1
# policy_param_max = 0
#
# [source_task, source_param, episodes_per_config, next_states_unclipped, actions_clipped] = sourceTaskCreation(episode_length, batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max)
#
# np.savetxt("source_task.csv", source_task, delimiter=",")
# np.savetxt("source_param.csv", source_param, delimiter=",")
# np.savetxt("episodes_per_config.csv", episodes_per_config, delimiter=",")
# np.savetxt("next_states_unclipped.csv", next_states_unclipped, delimiter=",")
# np.savetxt("actions_clipped.csv", actions_clipped, delimiter=",")
