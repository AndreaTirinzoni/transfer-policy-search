import gym
import numpy as np

def createEpisode(env, episode_length, param, state, variance_action):
    """
    The function creates a new episode

    Args:
        env: OpenAI environment
        episode_length: length of the episode
        param: policy parameter
        state: initial state
        variance_action: variance of the action distribution
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

def sourceTaskCreationWithReinforce(env, num_episodes, batch_size, discount_factor):
    """
    Creates the source dataset for IS and perform REINFORCE

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for
        batch_size: Number of episodes for each batch
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_disc_reward and episode_rewards related to the batch.
        A data structure containing all informations about [state, action reward] in all time steps.
        A data structure containing the parameters for all episode contained in source_task.
    """
    #param = np.random.normal(mean_initial_param, variance_initial_param)
    param = 0
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


    source_task = np.zeros((num_episodes, episode_length*3+1)) # every line a task, every task has all [state, action, reward]
    # Every line is a task, every task has [discounted_return, policy_parameter, A, B, variance] where A and B are the evironment params
    source_param = np.zeros((num_episodes, 5))

    for i_batch in range(num_batch):
        episode_informations = np.zeros((batch_size, 3))
        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            # Reset the environment and pick the first action
            state = env.reset()
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
            source_task[i_task, t*3+3] = episode[t, 3]

            episode_informations[i_episode,:] = [gradient_est, total_return, discounted_return]

            #I populate the source parameters
            source_param[i_task, 0] = discounted_return
            source_param[i_task, 1] = param
            source_param[i_task, 2] = env.A
            source_param[i_task, 3] = env.B
            source_param[i_task, 4] = env.sigma_noise**2
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
    env_param = np.linspace(env_param_min, env_param_max, 80)
    i_task = 0
    episodes_per_configuration = np.zeros(policy_param.shape[0]*env_param.shape[0])
    i_configuration = 0
    episode_per_param = batch_size
    length_source_task = policy_param.shape[0]*env_param.shape[0]*episode_per_param
    source_task = np.zeros((length_source_task, episode_length*3+1)) # every line a task, every task has all [state, action, reward]
    # Every line is a task, every task has [discounted_return, policy_parameter, A, B, variance, parameter_configuration id] where A and B are the evironment params
    source_param = np.zeros((length_source_task, 5))
    for i_policy_param in range(policy_param.shape[0]):
        for i_env_param in range(env_param.shape[0]):
            #env.setA(env_param[i_env_param])
            # Iterate for every episode
            for i_episode_param in range(episode_per_param):
            # Reset the environment and pick the first action
                state = env.reset()
                episode = createEpisode(env, episode_length, policy_param[i_policy_param], state, variance_action) # [state, action, reward, next_state]
                # Go through the episode and compute estimators

                discounted_return = np.sum(np.multiply(np.power(discount_factor*np.ones(episode.shape[0]), range(episode.shape[0])), episode[:, 2]))

                #I populate the source task
                source_task[i_task, 0::3] = np.concatenate((episode[:, 0].T, [episode[-1, 3]]))
                source_task[i_task, 1::3] = episode[:, 1]
                source_task[i_task, 2::3] = episode[:, 2]

                #I populate the source parameters
                source_param[i_task, 0] = discounted_return
                source_param[i_task, 1] = policy_param[i_policy_param]
                source_param[i_task, 2] = env.A
                source_param[i_task, 3] = env.B
                source_param[i_task, 4] = env.sigma_noise**2
                i_task += 1

            episodes_per_configuration[i_configuration] = episode_per_param
            i_configuration += 1


    return source_task, source_param, episodes_per_configuration.astype(int)

env = gym.make('LQG1D-v0')
episode_length = 50
mean_initial_param = 0
variance_initial_param = 0
variance_action = 0.001
np.random.seed(2000)
num_episodes=1000
batch_size=40
discount_factor = 0.99
env_param_min = -2
env_param_max = 2
policy_param_min = -1
policy_param_max = 0

[source_task, source_param, episodes_per_config] = sourceTaskCreation(env, episode_length, batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max)

np.savetxt("source_task.csv", source_task, delimiter=",")
np.savetxt("source_param.csv", source_param, delimiter=",")
np.savetxt("episodes_per_config.csv", episodes_per_config, delimiter=",")
