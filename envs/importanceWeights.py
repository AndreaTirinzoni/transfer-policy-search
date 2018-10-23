import gym
import math as m
import numpy as np
import algorithmPolicySearch as alg
import plotting as plt
from collections import namedtuple


def optimalPolicy(env, num_episodes, discount_factor):
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
    gradient_off_policy = np.zeros(source_task.shape[0])

    for i_episode in range(source_task.shape[0]):
        gradient_off_policy[i_episode] = sum(source_task[i_episode, 2::3] - param * source_task[i_episode, 1::3] * source_task[i_episode, 1::3] / variance_action)

    return gradient_off_policy

def computeImportanceWeightsSourceTarget(env, param, source_param, variance_action, source_task, episode_length):
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
    weights = np.ones(source_task.shape[0])

    for i_episode in range(source_task.shape[0]):
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
            # model_src = 1
            # model_tgt = 1

            weights[i_episode] = weights[i_episode] * policy_src/policy_tgt * model_src/model_tgt

        #print(weights[i_episode], i_episode)

    return weights

def offPolicyUpdate(env, param, source_param, episodes_per_config, source_task, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor):
    """

    :param env:
    :param param:
    :param source_param:
    :param episodes_per_config:
    :param source_task:
    :param variance_action:
    :param episode_length:
    :param batch_size:
    :param t:
    :param m_t:
    :param v_t:
    :param discount_factor:
    :return:
    """
    #Compute gradients of the source task
    gradient_off_policy = computeGradientsSourceTarget(param, source_task, variance_action)
    #Compute importance weights_source_target of source task
    weights_source_target = computeImportanceWeightsSourceTarget(env, param, source_param, variance_action, source_task, episode_length)
    # num_episodes_target = m.ceil((batch_size - 2*np.sum(weights_source_target) - m.sqrt(batch_size*(batch_size+4*(np.dot(weights_source_target, weights_source_target)-np.sum(weights_source_target)))))/2)
    num_episodes_target = batch_size
    episode_informations = np.zeros((num_episodes_target, 3))
    # Create new parameters and new tasks associated to episodes, used tu update the source_param and source_task later
    source_param_new = np.ones((num_episodes_target, 5))
    source_task_new = np.ones((num_episodes_target, episode_length*3+1))
    # Iterate for every episode in batch
    for i_episode in range(num_episodes_target):
        # Reset the environment and pick the first action
        print(i_episode)
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
        print(i_batch)
        stats.policy_parameter[i_batch] = param
        [source_param, source_task, episodes_per_config, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch] = offPolicyUpdate(env, param, source_param, episodes_per_config, source_task, variance_action, episode_length, batch_size, t, m_t, v_t, discount_factor)
        # Update statistics
        stats.episode_total_rewards[i_batch] = tot_reward_batch
        stats.episode_disc_rewards[i_batch] = discounted_reward_batch
    return stats


EpisodeStats = namedtuple("Stats",["episode_total_rewards", "episode_disc_rewards", "policy_parameter"])
np.set_printoptions(precision=4)
env = gym.make('LQG1D-v0')
#env = gym.make('LQG1D-v1')
eps = 10**-16
episode_length = 50
mean_initial_param = -1
variance_initial_param = 0.2
variance_action = 0.001
num_episodes=400
batch_size=20
num_batch = num_episodes//batch_size
discount_factor = 0.99
runs = 10

source_task = np.genfromtxt('source_task.csv', delimiter=',')
episodes_per_config = np.genfromtxt('episodes_per_config.csv', delimiter=',').astype(int)
source_param = np.genfromtxt('source_param.csv', delimiter=',')

discounted_reward_off_policy = np.zeros((runs, num_batch))
discounted_reward_reinfroce = np.zeros((runs, num_batch))
policy_param_off_policy = np.zeros((runs, num_batch))
policy_param_reinfroce = np.zeros((runs, num_batch))
for i_run in range(runs):
    print(i_run)
    np.random.seed(2000+500*i_run)
    initial_param = np.random.normal(mean_initial_param, variance_initial_param)
    off_policy = offPolicyImportanceSampling(env, batch_size, discount_factor, source_task, source_param, episodes_per_config, variance_action, episode_length, initial_param, num_batch)
    reinforce = alg.reinforce(env, num_episodes, batch_size, discount_factor, episode_length, initial_param)
    discounted_reward_off_policy[i_run,:] = off_policy.episode_disc_rewards
    discounted_reward_reinfroce[i_run, :] = reinforce.episode_disc_rewards
    policy_param_off_policy[i_run,:] = off_policy.policy_parameter
    policy_param_reinfroce[i_run, :] = reinforce.policy_parameter

np.savetxt("discounted_reward_off_policy_1.csv", discounted_reward_off_policy, delimiter=",")
np.savetxt("discounted_reward_reinfroce_1.csv", discounted_reward_reinfroce, delimiter=",")
np.savetxt("policy_param_off_policy_1.csv", policy_param_off_policy, delimiter=",")
np.savetxt("policy_param_reinfroce_1.csv", policy_param_reinfroce, delimiter=",")

# discounted_reward_off_policy = np.genfromtxt('discounted_reward_off_policy.csv', delimiter=',')
# discounted_reward_reinfroce = np.genfromtxt('discounted_reward_reinfroce.csv', delimiter=',')

stats_opt = optimalPolicy(env, num_episodes, discount_factor).episode_disc_rewards # Optimal policy

plt.plot_mean_and_variance(discounted_reward_reinfroce, discounted_reward_off_policy, stats_opt, num_batch, discount_factor)
plt.plot_mean_and_variance(policy_param_off_policy, policy_param_reinfroce, stats_opt, num_batch, discount_factor)
