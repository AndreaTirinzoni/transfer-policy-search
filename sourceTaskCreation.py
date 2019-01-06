import gym
import envs
import numpy as np
import math as m

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

    information_size = state_space_size+2+state_space_size+state_space_size+1+state_space_size
    batch = np.zeros((batch_size, episode_length, information_size)) #[state, clipped_action, reward, next_state, unclipped_state, action]
    trajectory_length = np.zeros(batch_size)
    for i_batch in range(batch_size):
        state = env.reset()

        for t in range(episode_length):
            # Take a step
            mean_action = np.sum(np.multiply(param, state))
            action = np.random.normal(mean_action, m.sqrt(variance_action))
            next_state, reward, done, unclipped_state, clipped_action, next_state_denoised = env.step(action)
            # Keep track of the transition
            # env.render()
            batch[i_batch, t, 0:state_space_size] = state
            batch[i_batch, t, state_space_size] = clipped_action
            batch[i_batch, t, state_space_size+1] = reward
            batch[i_batch, t, state_space_size+2:state_space_size+2+state_space_size] = next_state
            batch[i_batch, t, state_space_size+2+state_space_size:state_space_size+2+state_space_size+state_space_size] = unclipped_state
            batch[i_batch, t, state_space_size+2+state_space_size+state_space_size] = action
            batch[i_batch, t, state_space_size+2+state_space_size+state_space_size+1:state_space_size+2+state_space_size+state_space_size+1+state_space_size] = next_state_denoised

            if done:
                break

            state = next_state
        trajectory_length[i_batch] = t

    return batch, trajectory_length

def sourceTaskCreationAllCombinations(env, episode_length, batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size):
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
    :param linspace_policy: number of policies from policy_min to policy_max
    :param linspace_env: number of environment parameters from env_min to env_max
    :return:A data structure containing all informations about the episodes,
            a data structure containing informations about the parameters of
            the episodes and a vector containing the number of episodes for every configuration
    """

    policy_param = np.linspace(policy_param_min, policy_param_max, linspace_policy)
    env_param = np.linspace(env_param_min, env_param_max, linspace_env)
    i_episode = 0
    episodes_per_configuration = np.zeros(policy_param.shape[0]*env_param.shape[0])
    i_configuration = 0
    episode_per_param = batch_size
    length_source_task = policy_param.shape[0]*env_param.shape[0]*episode_per_param
    source_task = np.zeros((length_source_task, episode_length, state_space_size + 2 + state_space_size)) # every line a task, every task has all [clipped_state, action, reward]
    # Every line is a task, every task has [discounted_return, policy_parameter, env_params, variance]
    source_param = np.zeros((length_source_task, 1+param_space_size+env_param_space_size+1)) # discounted rewards, policy parameter, environment parameter, episode length
    next_states_unclipped = np.zeros((length_source_task, episode_length, state_space_size))
    next_states_unclipped_denoised = np.zeros((length_source_task, episode_length, state_space_size))
    actions_clipped = np.zeros((length_source_task, episode_length))

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    for i_policy_param in range(policy_param.shape[0]):

        for i_env_param in range(env_param.shape[0]):

            env.setParams(np.concatenate(([env_param[i_env_param]], np.ravel(env.B), [env.sigma_noise**2])))

            # Reset the environment and pick the first action
            [batch, trajectory_length] = createBatch(env, episode_per_param, episode_length, policy_param[i_policy_param], state_space_size, variance_action) # [state, action, reward, next_state]

            #  Go through the episode and compute estimators
            #I populate the source task
            discounted_return = np.sum((discount_factor_timestep * batch[:, :, state_space_size+1]), axis=1)

            #I populate the source task
            source_task[i_episode:i_episode+episode_per_param, :, 0:state_space_size] = batch[:, :, 0:state_space_size] #state
            source_task[i_episode:i_episode+episode_per_param, :, state_space_size] = batch[:, :, state_space_size+2+state_space_size+state_space_size] #unclipped action
            source_task[i_episode:i_episode+episode_per_param, :, state_space_size+1] = batch[:, :, state_space_size+1] #reward
            source_task[i_episode:i_episode+episode_per_param, :, state_space_size+2:state_space_size+2+state_space_size] = batch[:, :, state_space_size+2:state_space_size+2+state_space_size] #next state

            #unclipped next_states and actions
            next_states_unclipped[i_episode:i_episode+episode_per_param, :] = batch[:, :, state_space_size+2+state_space_size:state_space_size+2+state_space_size+state_space_size]
            next_states_unclipped_denoised[i_episode:i_episode+episode_per_param, :] = batch[:, :, state_space_size+2+state_space_size+state_space_size+1:state_space_size+2+state_space_size+state_space_size+1+state_space_size]
            actions_clipped[i_episode:i_episode+episode_per_param, :] = batch[:, :, state_space_size]

            #I populate the source parameters
            source_param[i_episode:i_episode+episode_per_param, 0] = discounted_return
            source_param[i_episode:i_episode+episode_per_param, 1:1+param_space_size] = policy_param[i_policy_param]
            source_param[i_episode:i_episode+episode_per_param, 1+param_space_size:1+param_space_size+env_param_space_size] = env.getEnvParam().T
            source_param[i_episode:i_episode+episode_per_param, 1+param_space_size+env_param_space_size] = trajectory_length

            i_episode += episode_per_param

            episodes_per_configuration[i_configuration] = episode_per_param
            i_configuration += 1


    return source_task, source_param, episodes_per_configuration.astype(int), next_states_unclipped, actions_clipped, next_states_unclipped_denoised

def sourceTaskCreationSpec(env, episode_length, batch_size, discount_factor, variance_action, policy_params, env_params, param_space_size, state_space_size, env_param_space_size):
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
    :param linspace_policy: number of policies from policy_min to policy_max
    :param linspace_env: number of environment parameters from env_min to env_max
    :return:A data structure containing all informations about the episodes,
            a data structure containing informations about the parameters of
            the episodes and a vector containing the number of episodes for every configuration
    """

    i_episode = 0
    episodes_per_configuration = np.zeros(policy_params.shape[0]*env_params.shape[0])
    i_configuration = 0
    episode_per_param = batch_size
    length_source_task = policy_params.shape[0]*env_params.shape[0]*episode_per_param
    source_task = np.zeros((length_source_task, episode_length, state_space_size + 2 + state_space_size)) # every line a task, every task has all [clipped_state, action, reward]
    # Every line is a task, every task has [discounted_return, policy_parameter, env_params, variance]
    source_param = np.zeros((length_source_task, 1+param_space_size+env_param_space_size+1))
    next_states_unclipped = np.zeros((length_source_task, episode_length, state_space_size))
    next_states_unclipped_denoised = np.zeros((length_source_task, episode_length, state_space_size))
    actions_clipped = np.zeros((length_source_task, episode_length))

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    for i_policy_param in range(policy_params.shape[0]):

        for i_env_param in range(env_params.shape[0]):

            env.setParams(env_params[i_env_param, :])

            # Reset the environment and pick the first action
            [batch, trajectory_length] = createBatch(env, episode_per_param, episode_length, policy_params[i_policy_param, :], state_space_size, variance_action) # [state, action, reward, next_state]

            #  Go through the episode and compute estimators

            discounted_return = np.sum((discount_factor_timestep * batch[:, :, state_space_size+1]), axis=1)

            #I populate the source task
            source_task[i_episode:i_episode+episode_per_param, :, 0:state_space_size] = batch[:, :, 0:state_space_size]
            source_task[i_episode:i_episode+episode_per_param, :, state_space_size] = batch[:, :, state_space_size+2+state_space_size+state_space_size]
            source_task[i_episode:i_episode+episode_per_param, :, state_space_size+1] = batch[:, :, state_space_size+1]
            source_task[i_episode:i_episode+episode_per_param, :, state_space_size+2:] = batch[:, :, state_space_size+2:state_space_size+2+state_space_size]

            #unclipped next_states and actions
            next_states_unclipped[i_episode:i_episode+episode_per_param, :] = batch[:, :, state_space_size+2+state_space_size:state_space_size+2+state_space_size+state_space_size]
            next_states_unclipped_denoised[i_episode:i_episode+episode_per_param, :] = batch[:, :, state_space_size+2+state_space_size+state_space_size+1:state_space_size+2+state_space_size+state_space_size+1+state_space_size]
            actions_clipped[i_episode:i_episode+episode_per_param, :] = batch[:, :, state_space_size]

            #I populate the source parameters
            source_param[i_episode:i_episode+episode_per_param, 0] = discounted_return
            source_param[i_episode:i_episode+episode_per_param, 1:1+param_space_size] = policy_params[i_policy_param, :]
            source_param[i_episode:i_episode+episode_per_param, 1+param_space_size:1+param_space_size+env_param_space_size] = env.getEnvParam().T
            source_param[i_episode:i_episode+episode_per_param, 1+param_space_size+env_param_space_size] = trajectory_length

            i_episode += episode_per_param

            episodes_per_configuration[i_configuration] = episode_per_param
            i_configuration += 1


    return source_task, source_param, episodes_per_configuration.astype(int), next_states_unclipped, actions_clipped, next_states_unclipped_denoised


def sourceTaskCreationMixture(env, episode_length, batch_size, discount_factor, variance_action, policy_params, episodes_per_config, n_config_cv, param_space_size, state_space_size, env_param_space_size):
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
    :param linspace_policy: number of policies from policy_min to policy_max
    :param linspace_env: number of environment parameters from env_min to env_max
    :return:A data structure containing all informations about the episodes,
            a data structure containing informations about the parameters of
            the episodes and a vector containing the number of episodes for every configuration
    """

    i_episode = 0
    number_of_iteration = 30
    episodes_per_configuration = np.zeros(number_of_iteration)
    i_configuration = 0
    length_source_task = number_of_iteration * batch_size
    source_task = np.zeros((length_source_task, episode_length, state_space_size + 2 + state_space_size)) # every line a task, every task has all [clipped_state, action, reward]
    # Every line is a task, every task has [discounted_return, policy_parameter, env_params, variance]
    source_param = np.zeros((length_source_task, 1+param_space_size+env_param_space_size+1))
    next_states_unclipped = np.zeros((length_source_task, episode_length, state_space_size))
    next_states_unclipped_denoised = np.zeros((length_source_task, episode_length, state_space_size))
    actions_clipped = np.zeros((length_source_task, episode_length))
    alpha_j_tr = episodes_per_config[n_config_cv:] / np.sum(episodes_per_config)
    alpha_tr = np.sum(alpha_j_tr)
    probabilities = alpha_j_tr / alpha_tr

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    for i in range(number_of_iteration):

        policy_param_index = np.random.choice(np.arange(0, probabilities.shape[0]), p=probabilities)
        # Reset the environment and pick the first action
        policy_param = policy_params[policy_param_index]
        [batch, trajectory_length] = createBatch(env, batch_size, episode_length, policy_param, state_space_size, variance_action) # [state, action, reward, next_state]

        #  Go through the episode and compute estimators

        discounted_return = np.sum((discount_factor_timestep * batch[:, :, state_space_size+1]), axis=1)

        #I populate the source task
        source_task[i_episode:i_episode+batch_size, :, 0:state_space_size] = batch[:, :, 0:state_space_size]
        source_task[i_episode:i_episode+batch_size, :, state_space_size] = batch[:, :, state_space_size+2+state_space_size+state_space_size]
        source_task[i_episode:i_episode+batch_size, :, state_space_size+1] = batch[:, :, state_space_size+1]
        source_task[i_episode:i_episode+batch_size, :, state_space_size+2:] = batch[:, :, state_space_size+2:state_space_size+2+state_space_size]

        #unclipped next_states and actions
        next_states_unclipped[i_episode:i_episode+batch_size, :] = batch[:, :, state_space_size+2+state_space_size:state_space_size+2+state_space_size+state_space_size]
        next_states_unclipped_denoised[i_episode:i_episode+batch_size, :] = batch[:, :, state_space_size+2+state_space_size+state_space_size+1:state_space_size+2+state_space_size+state_space_size+1+state_space_size]
        actions_clipped[i_episode:i_episode+batch_size, :] = batch[:, :, state_space_size]

        #I populate the source parameters
        source_param[i_episode:i_episode+batch_size, 0] = discounted_return
        source_param[i_episode:i_episode+batch_size, 1:1+param_space_size] = policy_param
        source_param[i_episode:i_episode+batch_size, 1+param_space_size:1+param_space_size+env_param_space_size] = env.getEnvParam().T
        source_param[i_episode:i_episode+batch_size, 1+param_space_size+env_param_space_size] = trajectory_length

        i_episode += batch_size

        episodes_per_configuration[i_configuration] = batch_size
        i_configuration += 1


    return source_task, source_param, episodes_per_configuration.astype(int), next_states_unclipped, actions_clipped, next_states_unclipped_denoised


#env = gym.make('LQG1D-v0')
# variance_action = 0.1
# episode_length = 20
# np.random.seed(2000)
# batch_size = 20
# discount_factor = 0.99
# env_param_min = 0.6
# env_param_max = 1.5
# policy_param_min = -1
# policy_param_max = -0.1
# linspace_env = 11
# linspace_policy = 10
# param_space_size = 1
# state_space_size = 1
# env_param_space_size = 3 #include variance as well
#
# [source_task, source_param, episodes_per_config, next_states_unclipped, actions_clipped, next_states_denoised] = sourceTaskCreationAllCombinations(env, episode_length, batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size)

# np.savetxt("source_task.csv", source_task, delimiter=",")
# np.savetxt("source_param.csv", source_param, delimiter=",")
# np.savetxt("episodes_per_config.csv", episodes_per_config, delimiter=",")
# np.savetxt("next_states_unclipped.csv", next_states_unclipped, delimiter=",")
# np.savetxt("actions_clipped.csv", actions_clipped, delimiter=",")
