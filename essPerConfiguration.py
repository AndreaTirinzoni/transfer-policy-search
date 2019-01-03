import numpy as np
import math as m
import sourceTaskCreation as stc
import gym

def getEpisodesInfoFromSource(source_dataset, env_param):

    param_policy_src = source_dataset.source_param[:, 1:1+env_param.param_space_size] # policy parameter of source
    state_t = source_dataset.source_task[:, :, 0:env_param.state_space_size] # state t
    state_t1 = source_dataset.next_states_unclipped # state t+1
    unclipped_action_t = source_dataset.source_task[:, :, env_param.state_space_size]
    clipped_actions = source_dataset.clipped_actions
    env_param_src = source_dataset.source_param[:, 1+env_param.param_space_size:1+env_param.param_space_size+env_param.env_param_space_size]
    trajectories_length = source_dataset.source_param[:, 1+env_param.param_space_size+env_param.env_param_space_size]

    return [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions, trajectories_length]


def computeMultipleImportanceWeightsSourceTarget(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions, trajectories_length] = getEpisodesInfoFromSource(source_dataset, env_param)
    variance_action = simulation_param.variance_action

    n = state_t.shape[0]
    param_indices = np.concatenate(([0], np.cumsum(np.delete(source_dataset.episodes_per_config, -1))))

    combination_src_parameters = param_policy_src[param_indices, :]#policy parameter of source not repeated
    combination_src_parameters_env = env_param_src[param_indices, :]#policy parameter of source not repeated

    evaluated_trajectories = source_dataset.source_distributions.shape[0]
    state_t1_denoised_current = env_param.env.stepDenoisedCurrent(state_t, clipped_actions)
    variance_env = env_param_src[:, -1] # variance of the model transition

    if batch_size != 0:

        state_t = np.repeat(state_t[:, :, :, np.newaxis], combination_src_parameters.shape[0], axis=3) # state t
        state_t1 = np.repeat(state_t1[:, :, :, np.newaxis], combination_src_parameters.shape[0], axis=3) # state t+1
        unclipped_action_t = np.repeat(unclipped_action_t[:, :, np.newaxis], combination_src_parameters.shape[0], axis=2) # action t
        clipped_actions = np.repeat(clipped_actions[:, :, np.newaxis], combination_src_parameters.shape[0], axis=2) # clipped action t
        state_t1_denoised = env_param.env.stepDenoised(combination_src_parameters_env, state_t, clipped_actions)

        policy_src_new_traj = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[evaluated_trajectories:, :, :] - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, :], state_t[evaluated_trajectories:, :, :, :]), axis=2))**2)/(2*variance_action))
        model_src_new_traj = 1/np.sqrt((2*m.pi*variance_env[evaluated_trajectories:, np.newaxis, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1[evaluated_trajectories:, :, :, :] - state_t1_denoised[evaluated_trajectories:, :, :, :])**2, axis=2) / (2*variance_env[evaluated_trajectories:, np.newaxis, np.newaxis]))

        policy_src_new_param = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[0:evaluated_trajectories, :, 0] - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, -1], state_t[0:evaluated_trajectories, :, :, 0]), axis=2))**2)/(2*variance_action))
        model_src_new_param = 1/np.sqrt((2*m.pi*variance_env[0:evaluated_trajectories, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1[0:evaluated_trajectories, :, :, 0] - state_t1_denoised_current[0:evaluated_trajectories, :, :])**2, axis=2) / (2*variance_env[0:evaluated_trajectories, np.newaxis]))

        mask_new_param = np.ones(policy_src_new_param.shape)
        mask_new_trajectories = np.ones(policy_src_new_traj.shape)
        for t in range(state_t.shape[1]):
            indices_new_trajectories = trajectories_length[evaluated_trajectories:] < t
            if indices_new_trajectories[indices_new_trajectories == "True"].shape[0] > 0:
                mask_new_trajectories[indices_new_trajectories, t, :] = 1/(policy_src_new_traj[indices_new_trajectories, t, :] * model_src_new_traj[indices_new_trajectories, t, :])

            indices_new_param = trajectories_length[0:evaluated_trajectories] < t
            if indices_new_param[indices_new_param == "True"].shape[0] > 0:
                mask_new_param[indices_new_param, t] = 1/(policy_src_new_param[indices_new_param, t, :] * model_src_new_param[indices_new_param, t, :])

        source_dataset.mask_weights = np.concatenate((source_dataset.mask_weights, mask_new_param[:, :, np.newaxis]), axis=2)
        source_dataset.mask_weights = np.concatenate((source_dataset.mask_weights, mask_new_trajectories), axis=0)

        policy_src_new_traj = np.prod(policy_src_new_traj * mask_new_trajectories, axis=1)
        model_src_new_traj = np.prod(model_src_new_traj, axis=1)

        policy_src_new_param = np.prod(policy_src_new_param * mask_new_param, axis=1)
        model_src_new_param = np.prod(model_src_new_param, axis=1)

        source_dataset.source_distributions = np.concatenate((source_dataset.source_distributions, np.matrix(policy_src_new_param * model_src_new_param).T), axis=1)
        source_dataset.source_distributions = np.concatenate((source_dataset.source_distributions, np.matrix(policy_src_new_traj * model_src_new_traj)), axis=0)

        policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[:, :, 0] - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t[:, :, :, 0]), axis=2))**2)/(2*variance_action))
        model_tgt = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1[:, :, :, 0] - state_t1_denoised_current)**2, axis=2) / (2*variance_env[:, np.newaxis]))

        for t in range(state_t.shape[1]):
            indices = trajectories_length < t
            policy_tgt[indices, t] = 1
            model_tgt[indices, t] = 1

        policy_tgt = np.prod(policy_tgt, axis=1)
        model_tgt = np.prod(model_tgt, axis=1)

    else:
        policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2))**2)/(2*variance_action))
        model_tgt = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1 - state_t1_denoised_current)**2, axis=2) / (2*variance_env[:, np.newaxis]))

        for t in range(state_t.shape[1]):
            indices = trajectories_length < t
            policy_tgt[indices, t] = 1
            model_tgt[indices, t] = 1

        policy_tgt = np.prod(policy_tgt, axis=1)
        model_tgt = np.prod(model_tgt, axis=1)

    mis_denominator = np.squeeze(np.asarray(np.sum(np.multiply(source_dataset.episodes_per_config[np.newaxis, :]/n, source_dataset.source_distributions), axis=1)))

    weights = policy_tgt * model_tgt / mis_denominator

    return [weights, mis_denominator]


def computeMultipleImportanceWeightsSourceTargetPerDecision(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions, trajectories_length] = getEpisodesInfoFromSource(source_dataset, env_param)
    variance_action = simulation_param.variance_action

    n = state_t.shape[0]
    param_indices = np.concatenate(([0], np.cumsum(np.delete(source_dataset.episodes_per_config, -1))))

    combination_src_parameters = param_policy_src[param_indices, :]#policy parameter of source not repeated
    combination_src_parameters_env = env_param_src[param_indices, :]#policy parameter of source not repeated

    evaluated_trajectories = source_dataset.source_distributions.shape[0]
    state_t1_denoised_current = env_param.env.stepDenoisedCurrent(state_t, clipped_actions)
    variance_env = env_param_src[:, -1] # variance of the model transition

    if batch_size != 0:

        state_t = np.repeat(state_t[:, :, :, np.newaxis], combination_src_parameters.shape[0], axis=3) # state t
        state_t1 = np.repeat(state_t1[:, :, :, np.newaxis], combination_src_parameters.shape[0], axis=3) # state t+1
        unclipped_action_t = np.repeat(unclipped_action_t[:, :, np.newaxis], combination_src_parameters.shape[0], axis=2) # action t
        clipped_actions = np.repeat(clipped_actions[:, :, np.newaxis], combination_src_parameters.shape[0], axis=2) # action t
        state_t1_denoised = env_param.env.stepDenoised(combination_src_parameters_env, state_t, clipped_actions)

        policy_src_new_traj = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[evaluated_trajectories:, :, :] - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, :], state_t[evaluated_trajectories:, :, :, :]), axis=2))**2)/(2*variance_action)), axis=1)
        model_src_new_traj = np.cumprod(1/np.sqrt((2*m.pi*variance_env[evaluated_trajectories:, np.newaxis, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1[evaluated_trajectories:, :, :] - state_t1_denoised[evaluated_trajectories:, :, :])**2, axis=2) / (2*variance_env[evaluated_trajectories:, np.newaxis, np.newaxis])), axis=1)

        policy_src_new_param = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[0:evaluated_trajectories, :, 0] - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, -1], state_t[0:evaluated_trajectories, :, :, 0]), axis=2))**2)/(2*variance_action)), axis=1)
        model_src_new_param = np.cumprod(1/np.sqrt((2*m.pi*variance_env[0:evaluated_trajectories, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1[0:evaluated_trajectories, :, :, 0] - state_t1_denoised_current[0:evaluated_trajectories, :, :])**2, axis=2) / (2*variance_env[0:evaluated_trajectories, np.newaxis])), axis=1)

        source_dataset.source_distributions = np.concatenate((source_dataset.source_distributions, (policy_src_new_param * model_src_new_param)[:, :, np.newaxis]), axis=2)
        source_dataset.source_distributions = np.concatenate((source_dataset.source_distributions, (policy_src_new_traj * model_src_new_traj)), axis=0)

        policy_tgt = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[:, :, 0] - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t[:, :, :, 0]), axis=2))**2)/(2*variance_action)), axis=1)
        model_tgt = np.cumprod(1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1[:, :, :, 0] - state_t1_denoised_current)**2, axis=2) / (2*variance_env[:, np.newaxis])), axis=1)

    else:

        variance_env = env_param_src[:, -1] # variance of the model transition

        policy_tgt = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2))**2)/(2*variance_action)), axis=1)
        model_tgt = np.cumprod(1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1 - state_t1_denoised_current)**2, axis=2) / (2*variance_env[:, np.newaxis])), axis=1)

    mis_denominator = np.sum(np.multiply(source_dataset.episodes_per_config[np.newaxis, np.newaxis, :]/n, source_dataset.source_distributions), axis=2)

    weights = policy_tgt * model_tgt / mis_denominator

    return [weights, mis_denominator]


def computeEss(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration):

    trajectories_length = getEpisodesInfoFromSource(source_dataset, env_param)[-1]
    n = trajectories_length.shape[0]
    # variance_env = env_param_src[:, -1]
    # state_t1_denoised_current = env_param.env.stepDenoisedCurrent(state_t, clipped_actions)
    # variance_action = simulation_param.variance_action
    #
    # if algorithm_configuration.pd == 0:
    #     policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2)))**2)/(2*variance_action))
    #     model_tgt = 1/np.sqrt(2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size * np.exp(-(np.sum((state_t1 - state_t1_denoised_current)**2, axis=2)) / (2*variance_env[:, np.newaxis]))
    #
    #     for t in range(state_t.shape[1]):
    #         indices = trajectories_length < t
    #         policy_tgt[indices, t] = 1
    #         model_tgt[indices, t] = 1
    #
    #     policy_tgt = np.prod(policy_tgt, axis=1)
    #     model_tgt = np.prod(model_tgt, axis=1)
    #
    #     mis_denominator = np.squeeze(np.asarray(np.sum(np.multiply(source_dataset.episodes_per_config[np.newaxis, :] / n, source_dataset.source_distributions), axis=1)))
    #
    # else:
    #     policy_tgt = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2)))**2)/(2*variance_action)), axis=1)
    #     model_tgt = np.cumprod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size * np.exp(-(np.sum((state_t1 - state_t1_denoised_current)**2, axis=2)) / (2*variance_env[:, np.newaxis])), axis=1)
    #     mis_denominator = np.sum(np.multiply(source_dataset.episodes_per_config[np.newaxis, np.newaxis, :] / n, source_dataset.source_distributions), axis=2)

    weights = algorithm_configuration.computeWeights(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, 0)

    #if ess_den == 0:
        #print("problem")
    if algorithm_configuration.pd == 0:
        ess_den = np.sum(weights ** 2, axis=0)
        ess = np.sum(weights, axis=0) * n / ess_den
        min_index = 0
    else:
        ess = np.zeros(weights.shape[1])
        for t in range(weights.shape[1]):
            indices = trajectories_length >= t
            weights_timestep = weights[indices, t]
            ess_den = np.sum(weights_timestep ** 2, axis=0)
            ess[t] = np.sum(weights_timestep, axis=0) * n / ess_den

        min_index = np.argmin(ess)
        ess = ess[min_index]

    return [ess, min_index]


def computeEssSecond(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration):

    trajectories_length = getEpisodesInfoFromSource(source_dataset, env_param)[-1]
    n = trajectories_length.shape[0]

    weights = algorithm_configuration.computeWeights(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, 0)

    #if ess_den == 0:
        #print("problem")
    if algorithm_configuration.pd == 0:
        variance_weights = np.var(weights, axis=0)
        ess = n / (1 + variance_weights)
        min_index = 0
    else:
        ess = np.zeros(weights.shape[1])
        for t in range(weights.shape[1]):
            indices = trajectories_length >= t
            weights_timestep = weights[indices, t]
            variance_weights = np.var(weights_timestep, axis=0)
            ess[t] = n / (1 + variance_weights)

        min_index = np.argmin(ess)
        ess = ess[min_index]

    return [ess, min_index]


def essPerTarget(env_param_min, env_param_max, policy_param_min, policy_param_max, source_dataset, simulation_param, algorithm_configuration):
    """
    The function computes eh ess for every combination of environment_parameter and policy_parameter
    :param variance_action: variance of the action distribution
    :param env_param_min: minimum value assumed by the environment parameter
    :param env_param_max: maximum value assumed by the environment parameter
    :param policy_param_min: minimum value assumed by the policy parameter
    :param policy_param_max: maximum value assumed by the policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param episode_length: lenght of the episodes
    :return: A matrix containing ESS for every env_parameter - policy_parameter combination w.r.t the source task dataset
    """

    policy_param = np.linspace(policy_param_min, policy_param_max, 40)
    env_param = np.linspace(env_param_min, env_param_max, 160)
    ess1 = np.zeros((env_param.shape[0], policy_param.shape[0]))
    ess2 = np.zeros((env_param.shape[0], policy_param.shape[0]))
    for i_policy_param in range(policy_param.shape[0]):
        for i_env_param in range(env_param.shape[0]):
            [ess1, min_index1] = computeEss(policy_param[i_policy_param], env_param[i_env_param], source_dataset, simulation_param, algorithm_configuration)
            if algorithm_configuration.pd == 0:
                ess1[i_env_param, i_policy_param] = ess1
            else:
                ess1[i_env_param, i_policy_param] = ess1[min_index1]
            [ess2, min_index2] = computeEssSecond(policy_param[i_policy_param], env_param[i_env_param], source_dataset, simulation_param, algorithm_configuration)
            if algorithm_configuration.pd == 0:
                ess2[i_env_param, i_policy_param] = ess2
            else:
                ess2[i_env_param, i_policy_param] = ess2[min_index2]

    return [ess1, ess2]


env_src = gym.make("LQG1D-v0")
param_space_size = 1
state_space_size = 1
env_param_space_size = 3
episode_length = 20

mean_initial_param = 0
variance_initial_param = 0
variance_action = 0.1

batch_size = 20
discount_factor = 0.99

env_param_min = 0.5
env_param_max = 1.5
policy_param_min = -1
policy_param_max = -0.1
linspace_env = 22
linspace_policy = 20

[source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationAllCombinations(env_src, episode_length, batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size)

# print("Loading files")
# source_task = np.genfromtxt('source_task.csv', delimiter=',')
# episodes_per_config = np.genfromtxt('episodes_per_config.csv', delimiter=',').astype(int)
# source_param = np.genfromtxt('source_param.csv', delimiter=',')

print("Computing ESS")
[ess1, ess2] = essPerTarget(env_param_min, env_param_max, policy_param_min, policy_param_max, source_dataset, simulation_param, algorithm_configuration)

np.savetxt("ess_version1.csv", ess1, delimiter=",")
np.savetxt("ess_version2.csv", ess2, delimiter=",")
