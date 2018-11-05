import gym
import envs
import numpy as np
import math as m

def createBatch(env, batch_size, episode_length, param, variance_action):
    """
    The function creates a new episode

    Args:
        env: OpenAI environment
        episode_length: length of the episode
        batch_sizelength: size of the batch
        param: policy parameter
        state: initial state
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

def sourceTaskCreation(env, episode_length, batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max):
    """
    Creates the source dataset for IS

    Args:
        env: OpenAI environment.
        batch_size: Number of episodes for each batch
        discount_factor: Time-discount factor
        source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
        source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
        env_param_min: the minimum value of the environment parameter in our source_task
        env_param_max: the maximum value of the environment parameter in our source_task
        policy_param_min: the minimum value of the policy parameter in our source_task
        policy_param_max: the maximum value of the policyparameter in our source_task

    Returns:
        A data structure containing all informations about [state, action reward] in all time steps.
        A data structure containing the parameters for all episode contained in source_task.
        A data structure containing the number of episodes per environment_parameter - policy_parameter configuration
    """
    policy_param = np.linspace(policy_param_min, policy_param_max, 20)
    env_param = np.linspace(env_param_min, env_param_max, 40)
    i_episode = 0
    episodes_per_configuration = np.zeros(policy_param.shape[0]*env_param.shape[0])
    i_configuration = 0
    episode_per_param = batch_size
    length_source_task = policy_param.shape[0]*env_param.shape[0]*episode_per_param
    source_task = np.zeros((length_source_task, episode_length*3+1)) # every line a task, every task has all [state, action, reward]
    # Every line is a task, every task has [discounted_return, policy_parameter, A, B, variance, parameter_configuration id] where A and B are the evironment params
    source_param = np.zeros((length_source_task, 5))

    discount_factor_timestep = np.power(discount_factor*np.ones(episode_length), range(episode_length))

    for i_policy_param in range(policy_param.shape[0]):

        for i_env_param in range(env_param.shape[0]):

            #env.setA(env_param[i_env_param])

            # Reset the environment and pick the first action
            batch = createBatch(env, episode_per_param, episode_length, policy_param[i_policy_param], variance_action) # [state, action, reward, next_state]

            #  Go through the episode and compute estimators

            discounted_return = np.sum((discount_factor_timestep * batch[:, :, 2]), axis=1)

            #I populate the source task
            source_task[i_episode:i_episode+episode_per_param, 0::3] = np.concatenate((batch[:, :, 0], np.matrix(batch[:, -1, 3]).T), axis=1)
            source_task[i_episode:i_episode+episode_per_param, 1::3] = batch[:, :, 1]
            source_task[i_episode:i_episode+episode_per_param, 2::3] = batch[:, :, 2]

            #I populate the source parameters
            source_param[i_episode:i_episode+episode_per_param, 0] = discounted_return
            source_param[i_episode:i_episode+episode_per_param, 1] = policy_param[i_policy_param]
            source_param[i_episode:i_episode+episode_per_param, 2] = env.A
            source_param[i_episode:i_episode+episode_per_param, 3] = env.B
            source_param[i_episode:i_episode+episode_per_param, 4] = env.sigma_noise**2
            i_episode += episode_per_param

            episodes_per_configuration[i_configuration] = episode_per_param
            i_configuration += 1


    return source_task, source_param, episodes_per_configuration.astype(int)

env = gym.make('LQG1D-v0')
episode_length = 20
variance_action = 0.1
np.random.seed(2000)
num_episodes=1000
batch_size = 10
discount_factor = 0.99
env_param_min = 0.5
env_param_max = 2
policy_param_min = -1
policy_param_max = 0

[source_task, source_param, episodes_per_config] = sourceTaskCreation(env, episode_length, batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max)

np.savetxt("source_task.csv", source_task, delimiter=",")
np.savetxt("source_param.csv", source_param, delimiter=",")
np.savetxt("episodes_per_config.csv", episodes_per_config, delimiter=",")
