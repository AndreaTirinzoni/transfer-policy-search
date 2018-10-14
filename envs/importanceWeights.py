import gym
import itertools
import math as m
import collections
import numpy as np
import sys
import plotting as plot
import algorithmPolicySearch as alg

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

def sourceTaskCreationWithReinforce(env, num_episodes, batch_size, discount_factor, source_task, source_param):
    """
    Creates the source dataset for IS and perform REINFORCE

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for
        batch_size: Number of episodes for each batch
        discount_factor: Time-discount factor
        source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
        source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]

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

def sourceTaskCreation(env, num_episodes, num_batch, discount_factor):
    """
    Creates the source dataset for IS

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for
        batch_size: Number of episodes for each batch
        discount_factor: Time-discount factor
        source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
        source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]

    Returns:
        A data structure containing all informations about [state, action reward] in all time steps.
        A data structure containing the parameters for all episode contained in source_task.
    """
    policy_param = np.linspace(-1, 0, 20)
    env_param = np.linspace(-2, 2, 80)
    i_task = 0
    episode_per_param = num_batch
    length_source_task = policy_param.shape[0]*env_param.shape[0]*episode_per_param
    source_task = np.zeros((length_source_task, episode_length*3+1)) # every line a task, every task has all [state, action, reward]
    # Every line is a task, every task has [discounted_return, policy_parameter, A, B, variance] where A and B are the evironment params
    source_param = np.zeros((length_source_task, 5))
    for i_policy_param in range(policy_param.shape[0]):
        for i_env_param in range(env_param.shape[0]):
            episode_informations = np.zeros((episode_per_param, 3))
            #env.setA(env.param[i_env_param])
            # Iterate for every episode
            for i_episode_param in range(episode_per_param):
            # Reset the environment and pick the first action
                state = env.reset()
                episode = np.zeros((episode_length, 4)) # [state, action, reward, next_state]
                total_return = 0
                discounted_return = 0
                gradient_est = 0
                episode = createEpisode(env, episode_length, policy_param[i_policy_param], state) # [state, action, reward, next_state]

                # Go through the episode and compute estimators
                for t in range(episode.shape[0]):
                    # The return after this timestep
                    #total_return += episode[t, 2]
                    discounted_return += discount_factor ** t * episode[t, 2]

                    #I populate the source task
                    source_task[i_task, t*3] = episode[t, 0]
                    source_task[i_task, t*3+1] = episode[t, 1]
                    source_task[i_task, t*3+2] = episode[t, 2]
                source_task[i_task, t*3+3] = episode[t, 3]

                #I populate the source parameters
                source_param[i_task, 0] = discounted_return
                source_param[i_task, 1] = policy_param[i_policy_param]
                source_param[i_task, 2] = env.A
                source_param[i_task, 3] = env.B
                source_param[i_task, 4] = env.sigma_noise**2
                i_task += 1

    return source_task, source_param, episode_per_param

def computeGradientsSourceTarget(param, source_task, num_episodes, variance_action):
    """
        Compute the gradients estimation of the source targets with current policy

        Args:
            params: policy parameters
            source_task: source tasks
            num_episodes: number of episodes
            variance_action: variance of the action's distribution

        Returns:
            A vector containing all the gradients
    """
    gradient_off_policy = np.zeros(num_episodes)

    for i_episode in range(gradient_off_policy.shape[0]):
        gradient_off_policy[i_episode] = sum(source_task[i_episode, 2::3] - param * source_task[i_episode, 1::3] * source_task[i_episode, 1::3] / variance_action)

    return gradient_off_policy

def computeImportanceWeightsSourceTarget(env, param, source_param, num_episodes, variance_action, source_task, episode_length):
    """
        Compute the importance weights considering policy and transition model_src

        Args:
            env: OpenAI environment
            param: current policy parameter
            source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
            num_episodes: number of episodes
            variance_action: variance of the action's distribution
            source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
            episode_length: length of the episodes

        Returns:
            Returns the weights of the importance sampling
    """
    weights = np.ones(num_episodes)

    for i_episode in range(weights.shape[0]):
        for t in range(episode_length):
            param_policy = source_param[i_episode, 1] #policy parameter of source
            state_t = source_task[i_episode, t*3] # state t
            state_t1 = source_task[i_episode, t*3+3] # state t+1
            action_t = source_task[i_episode, t*3+1] # action t
            variance_env = source_param[i_episode, 4] # variance of the model transition
            A = source_param[i_episode, 2] # environment parameter A of src
            B = source_param[i_episode, 3] # environment parameter B of src
            policy_src = 1/m.sqrt(2*m.pi*variance_action) * m.exp(-(action_t - param*state_t)**2/(2*variance_action))
            policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * m.exp(-(action_t - param_policy*state_t)**2/(2*variance_action))
            model_src = 1/m.sqrt(2*m.pi*variance_env) * m.exp(-(state_t1 - env.A * state_t - env.B * action_t)**2/(2*variance_env))
            model_tgt = 1/m.sqrt(2*m.pi*variance_env) * m.exp(-(state_t1 - A * state_t - B * action_t)**2/(2*variance_env))
            if model_src==0 or model_tgt==0:
                print(model_src, model_tgt, param_policy, t)
            # model_src = 1
            # model_tgt = 1

            weights[i_episode] = weights[i_episode] * policy_src/policy_tgt * model_src/model_tgt

        #print(weights[i_episode], i_episode)

    return weights

def computeESS():
    ess = np.linalg.norm(weights_source_target_update, 1)**2 / np.linalg.norm(weights_source_target_update, 2)**2
    return ess

def offPolicyImportanceSampling(env, num_episodes, batch_size, discount_factor, source_task, source_param, variance_action, episode_length, mean_initial_param, episode_per_param):
    """
        Perform transfer from source tasks, using REINFORCE with IS

        Args:
            env: OpenAI environment
            num_episodes: number of episodes
            batch_size: size of the batch
            discount_factor: the discout factor
            source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
            source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
            variance_action: variance of the action's distribution
            episode_length: length of the episodes
            mean_initial_param: mean initial policy parameter

        Returns:
            An EpisodeStats object with two numpy arrays for episode_disc_reward and episode_rewards.

    """
    #param = np.random.normal(mean_initial_param, variance_initial_param)
    param = mean_initial_param
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Keeps track of useful statistics#
    stats = plot.EpisodeStats(
        episode_total_rewards=np.zeros(num_batch),
        episode_disc_rewards=np.zeros(num_batch))


    #Compute gradients of the source task
    gradient_off_policy = computeGradientsSourceTarget(param, source_task, num_episodes, variance_action)
    #Compute importance weights_source_target of source task
    weights_source_target = computeImportanceWeightsSourceTarget(env, param, source_param, num_episodes, variance_action, source_task, episode_length)

    for i_batch in range(num_batch):

        episode_informations = np.zeros((batch_size, 3))
        # Create new parameters and new tasks associated to episodes, used tu update the source_param and source_task later
        source_param_new = np.ones((batch_size, 5))
        source_task_new = np.ones((batch_size, episode_length*3+1))

        # Iterate for every episode in batch
        for i_episode in range(batch_size):
            # Reset the environment and pick the first action
            state = env.reset()
            total_return = 0
            discounted_return = 0
            gradient_est = 0
            episode = createEpisode(env, episode_length, param, state) # [state, action, reward, next_state]

            # Go through the episode and compute estimators
            for t in range(episode.shape[0]):
                # The return after this timestep
                total_return += episode[t, 2]
                discounted_return += discount_factor ** t * episode[t, 2]
                gradient_est += (episode[t, 1] - param * episode[t, 0]) * episode[t, 0] / variance_action
                source_task_new[i_episode, t*3] = episode[t, 0]
                source_task_new[i_episode, t*3+1] = episode[t, 1]
                source_task_new[i_episode, t*3+2] = episode[t, 2]
            source_task_new[i_episode, t*3+3] = episode[t, 3]

            episode_informations[i_episode,:] = [gradient_est, total_return, discounted_return]
            source_param_new[i_episode, 0] = discounted_return
            source_param_new[i_episode, 1] = param
            source_param_new[i_episode, 2] = env.A
            source_param_new[i_episode, 3] = env.B
            source_param_new[i_episode, 4] = env.sigma_noise**2


        #Update the parameters
        N = weights_source_target.shape[0] + batch_size
        weights_source_target_update = np.concatenate([weights_source_target, np.ones(batch_size)], axis=0) # are the weights used for computing ESS
        gradient_off_policy_update = np.concatenate([gradient_off_policy, episode_informations[:,0]], axis=0)
        discounted_rewards_all = np.concatenate([source_task[:,1], episode_informations[:,2]], axis=0)
        gradient = 1/N * np.dot(np.multiply(weights_source_target_update, gradient_off_policy_update), discounted_rewards_all)
        param, t, m_t, v_t = adam(param, -gradient, t, m_t, v_t, alpha=0.01)

        #Compute EFFECTIVE SAMPLE SIZE
        ess = computeESS()
        ess = np.linalg.norm(weights_source_target_update, 1)**2 / np.linalg.norm(weights_source_target_update, 2)**2
        #print(ess, N)

        #Compute rewards of batch
        tot_reward_batch = np.mean(episode_informations[:,1])
        discounted_reward_batch = np.mean(episode_informations[:,2])

        # Update weights_source_target, the gradients and source tasks and parameters with new episodes
        gradient_off_policy_new = computeGradientsSourceTarget(param, source_task_new, batch_size, variance_action)
        weights_source_target_new = computeImportanceWeightsSourceTarget(env, param, source_param_new, batch_size, variance_action, source_task_new, episode_length)

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

np.set_printoptions(precision=4)
env1 = gym.make('LQG1D-v0')
env2 = gym.make('LQG1D-v0')
eps = 10**-16
episode_length = 50
mean_initial_param = 0
variance_initial_param = 0
variance_action = 0.001
np.random.seed(2000)
num_episodes=1000
batch_size=40
num_batch = num_episodes//batch_size
discount_factor = 0.99

#[stats, source_task, source_param] = sourceTaskCreationWithReinforce(env1, num_episodes, batch_size, discount_factor)

[source_task, source_param, episode_per_param] = sourceTaskCreation(env1, num_episodes, num_batch, discount_factor)
stats_off_policy = offPolicyImportanceSampling(env2, num_episodes, batch_size, discount_factor, source_task, source_param, variance_action, episode_length, mean_initial_param, episode_per_param)
stats_opt = optimalPolicy(env2, num_episodes, batch_size, discount_factor) # Optimal policy
stats = alg.reinforce(env2, num_episodes, batch_size, discount_factor, episode_length, mean_initial_param, variance_initial_param) #reinforce without transfer

# Compare the statistics on_policy off_policy
plot.plot_algorithm_comparison_disc_two(stats, stats_off_policy, stats_opt, num_batch, discount_factor)
