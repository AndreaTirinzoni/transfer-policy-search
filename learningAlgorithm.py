import math as m
import numpy as np
import algorithmPolicySearch as alg
import random
import re
import simulationClasses as sc
import time

class BatchStats:

    def __init__(self, num_batch, param_space_size):

        self.total_rewards = np.zeros(num_batch)
        self.disc_rewards = np.zeros(num_batch)
        self.policy_parameter = np.zeros((num_batch, param_space_size))
        self.gradient = np.zeros((num_batch, param_space_size))
        self.ess = np.zeros(num_batch)
        self.n_def = np.zeros(num_batch)


class AlgorithmConfiguration:

    def __init__(self, estimator, cv, pd, baseline, approximation, adaptive, computeWeights, computeGradientUpdate, computeEss):

        self.estimator = estimator
        self.cv = cv
        self.pd = pd
        self.baseline = baseline
        self.approximation = approximation
        self.adaptive = adaptive
        self.computeWeights = computeWeights
        self.computeEss = computeEss
        self.computeGradientUpdate = computeGradientUpdate
        self.off_policy = None
        self.multid_approx = None
        self.dicrete_estimation = None
        self.model_estimation = None
        self.model_estimator = None


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


def computeGradientsSourceTargetTimestep(param, source_dataset, variance_action, env_param):

    state_t = source_dataset.source_task[:, :, 0:env_param.state_space_size] # state t
    action_t = source_dataset.source_task[:, :, env_param.state_space_size]

    gradient_off_policy = (action_t - np.sum(np.multiply(param[np.newaxis, np.newaxis, :],  state_t), axis=2))[:, :, np.newaxis] * state_t / variance_action

    return gradient_off_policy


def computeNdef(min_index, param, env_param, source_dataset, simulation_param, algorithm_configuration):

    trajectories_length = getEpisodesInfoFromSource(source_dataset, env_param)[-1]
    if algorithm_configuration.adaptive == "Yes":
        weights = algorithm_configuration.computeWeights(param, env_param, source_dataset, simulation_param, algorithm_configuration, 1, compute_n_def=1)[0]

    n = source_dataset.source_task.shape[0]
    if algorithm_configuration.pd == 1:
        #indices = trajectories_length >= min_index
        weights = weights[:, min_index]

    variance_weights = 1/n * np.sum((weights-1)**2)
    w_1 = np.linalg.norm(weights, 1)
    w_2 = np.linalg.norm(weights, 2)
    num_episodes_target1 = int(max(0, np.ceil((simulation_param.ess_min * w_1 / n) - (w_1 * n / (w_2 ** 2)))-1))

    c = (np.mean(weights**3) + 3*(1-np.mean(weights)))/(1 + variance_weights)**2
    num_episodes_target2 = np.ceil((simulation_param.ess_min - n / (1 + variance_weights))/(min(1, c)))
    num_episodes_target2 = int(np.clip(num_episodes_target2, 0, simulation_param.ess_min-simulation_param.defensive_sample))

    delta = 0.1

    if variance_weights < n * delta * (np.mean(weights) - 1)**2:
        num_episodes_target2 = simulation_param.ess_min - simulation_param.defensive_sample

    return [num_episodes_target1, num_episodes_target2]

# Compute gradients

def onlyGradient(algorithm_configuration, weights_source_target_update, gradient_off_policy_update, discounted_rewards_all, N):
    if(algorithm_configuration.pd == 0):
        gradient = 1/N * np.sum((np.squeeze(np.array(weights_source_target_update))[:, np.newaxis] * gradient_off_policy_update) * np.sum(discounted_rewards_all, axis=1)[:, np.newaxis], axis=0)
    else:
        gradient = 1/N * np.sum(np.sum((np.squeeze(np.array(weights_source_target_update))[:, :, np.newaxis] * gradient_off_policy_update) * discounted_rewards_all[:, :, np.newaxis], axis=1), axis=0)

    return gradient


def regressionFitting(y, x, n_config_cv, baseline_flag):
    #n_config_cv = min(n_config_cv, 100)
    # if baseline_flag == 1:
    #     baseline = np.squeeze(np.asarray(x[:, -1]))[:, np.newaxis]
    #     #x = np.concatenate([x, baseline], axis=1)
    #     x = np.concatenate([x[:, 0:n_config_cv], baseline], axis=1)
    # else:
    #     x = x[:, 0:n_config_cv]

    # train_size = int(np.ceil(x.shape[0]/4*3))
    # train_index = random.sample(range(x.shape[0]), train_size)
    # x_train = x[train_index]
    # y_train = y[train_index]
    # x_test = np.delete(x, train_index, axis=0)
    # y_test = np.delete(y, train_index, axis=0)
    # beta = np.squeeze(np.asarray(np.matmul(np.linalg.pinv(x_train[:, 1:]), y_train)))
    # error = y_test - np.dot(x_test[:, 1:], beta)

    beta = np.squeeze(np.asarray(np.matmul(np.linalg.pinv(x[:, 1:]), y)))
    error = y - np.dot(x[:, 1:], beta)

    # x_avg = np.squeeze(np.asarray(np.mean(x, axis=0)))
    # beta = np.matmul(np.linalg.inv(np.matmul((x[:, 1:]-x_avg[1:]).T, (x[:, 1:]-x_avg[1:]))), np.matmul((x[:, 1:]-x_avg[1:]).T, (y-y_avg)).T)
    # I = np.identity(y.shape[0])
    # error = np.squeeze(np.asarray(np.matmul(I - np.matmul(x[:, 1:], np.linalg.pinv(x[:, 1:])), y)))

    return np.mean(error)


def gradientAndRegression(algorithm_configuration, weights_source_target_update, gradient_off_policy_update, discounted_rewards_all, control_variates, n_config_cv, source_dataset, env_param):
    # Compute the gradient to fit according to the estimator properties
    if algorithm_configuration.pd == 1:
        # The algorithm is pd
        gradient_estimation = weights_source_target_update[:, :, np.newaxis] * gradient_off_policy_update * discounted_rewards_all[:, :, np.newaxis]
    else:
        gradient_estimation = weights_source_target_update[:, np.newaxis] * gradient_off_policy_update * np.sum(discounted_rewards_all, axis=1)[:, np.newaxis]

    #Fitting the regression
    if algorithm_configuration.pd == 1 and algorithm_configuration.approximation == 0:
        # Per decision and no approximation, I fit a regression for every timestep and sum
        gradient = np.zeros(gradient_estimation.shape[2])
        trajectories_length = getEpisodesInfoFromSource(source_dataset, env_param)[-1]
        for i in range(gradient_estimation.shape[2]):
        # Fitting the regression for every t 0...T-1
            for t in range(control_variates.shape[1]):
                #indices = trajectories_length >= t
                gradient[i] += regressionFitting(gradient_estimation[:, t, i], control_variates[:, t, :, i], n_config_cv, algorithm_configuration.baseline) #always the same, only the MIS with CV changes format of the x_avg array

    else:
        # No per decision or there is an approximation for the time direction, I some over it
        if algorithm_configuration.pd == 1:
            # Sum over time direction
            gradient_estimation = np.sum(gradient_estimation, axis=1)

        gradient = np.zeros(gradient_estimation.shape[1])
        for i in range(gradient_estimation.shape[1]):
            gradient[i] = regressionFitting(gradient_estimation[:, i], control_variates[:, :, i], n_config_cv, algorithm_configuration.baseline)

    return gradient


def gradientPolicySearch(algorithm_configuration, weights_source_target_update, gradient_off_policy_update, discounted_rewards_all, N):

    if algorithm_configuration.pd == 0:
        n = weights_source_target_update[weights_source_target_update == 1].shape[0]
        baseline = 0
        if algorithm_configuration.baseline == 1:
            baseline = np.sum(np.multiply((weights_source_target_update[:, np.newaxis] * gradient_off_policy_update)**2, (weights_source_target_update * np.sum(discounted_rewards_all, axis=1))[:, np.newaxis]), axis=0) / np.sum((weights_source_target_update[:, np.newaxis] * gradient_off_policy_update)**2, axis=0)
        gradient = 1/n * np.sum((np.squeeze(np.array(weights_source_target_update))[:, np.newaxis] * gradient_off_policy_update) * (np.sum(discounted_rewards_all, axis=1)[:, np.newaxis]-baseline), axis=0)

    else:
        n = weights_source_target_update[weights_source_target_update == 1].shape[0] / weights_source_target_update.shape[1]
        baseline = 0
        if algorithm_configuration.baseline == 1:
            baseline_den = np.sum(weights_source_target_update[:, :, np.newaxis] * gradient_off_policy_update**2, axis=0)
            baseline = np.sum(((weights_source_target_update[:, :, np.newaxis] * gradient_off_policy_update)**2) * discounted_rewards_all[:, :, np.newaxis], axis=0) / baseline_den
        gradient = 1/n * np.sum(np.sum((weights_source_target_update[:, :, np.newaxis] * gradient_off_policy_update) * (discounted_rewards_all[:, :, np.newaxis]-baseline[np.newaxis, :, :]), axis=1), axis=0)

    return gradient


#Compute weights of different estimators

def getEpisodesInfoFromSource(source_dataset, env_param):

    param_policy_src = source_dataset.source_param[:, 1:1+env_param.param_space_size] # policy parameter of source
    state_t = source_dataset.source_task[:, :, 0:env_param.state_space_size] # state t
    state_t1 = source_dataset.next_states_unclipped # state t+1
    unclipped_action_t = source_dataset.source_task[:, :, env_param.state_space_size]
    clipped_actions = source_dataset.clipped_actions
    env_param_src = source_dataset.source_param[:, 1+env_param.param_space_size:1+env_param.param_space_size+env_param.env_param_space_size]
    trajectories_length = source_dataset.source_param[:, 1+env_param.param_space_size+env_param.env_param_space_size]

    return [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions, trajectories_length]


def weightsPolicySearch(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size):


    if algorithm_configuration.pd == 0:
        weights = np.zeros(source_dataset.source_task.shape[0]-simulation_param.batch_size)
        weights = np.concatenate((weights, np.ones(simulation_param.batch_size)), axis=0)
    else:
        weights = np.zeros((source_dataset.source_task.shape[0]-simulation_param.batch_size, env_param.episode_length))
        weights = np.concatenate((weights, np.ones((simulation_param.batch_size, env_param.episode_length))), axis=0)

    return [weights, 0]


def computeImportanceWeightsSourceTarget(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size, compute_n_def=0):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_action_t, trajectories_length] = getEpisodesInfoFromSource(source_dataset, env_param)
    variance_action = simulation_param.variance_action
    variance_env = env_param_src[:, -1]
    param_indices = np.concatenate(([0], np.cumsum(np.delete(source_dataset.episodes_per_config, -1))))

    if algorithm_configuration.dicrete_estimation == 1 or algorithm_configuration.model_estimation == 0:
        state_t1_denoised_current = env_param.env.stepDenoisedCurrent(state_t, clipped_action_t)
    else:
        state_t1_denoised_current = algorithm_configuration.model_estimator.transition(state_t, clipped_action_t)

    state_t1_denoised = source_dataset.next_states_unclipped_denoised
    state_t1_denoised[source_dataset.initial_size:, :, :] = state_t1_denoised_current[source_dataset.initial_size:, :, :]

    if algorithm_configuration.pd == 0:

        policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2)))**2)/(2*variance_action))
        policy_src = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(param_policy_src[:, np.newaxis, :], state_t), axis=2)))**2)/(2*variance_action))

        model_tgt = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-(np.sum((state_t1 - state_t1_denoised_current)**2, axis=2)) / (2*variance_env[:, np.newaxis]))
        model_src = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-(np.sum((state_t1 - state_t1_denoised)**2, axis=2)) / (2*variance_env[:, np.newaxis]))

        mask = trajectories_length[:, np.newaxis] < np.repeat(np.arange(0, state_t.shape[1])[np.newaxis,:], repeats= state_t.shape[0], axis=0)
        policy_tgt[mask] = 1
        policy_src[mask] = 1
        model_tgt[mask] = 1
        model_src[mask] = 1

        policy_tgt = np.prod(policy_tgt, axis=1)
        policy_src = np.prod(policy_src, axis=1)

        model_tgt = np.prod(model_tgt, axis=1)
        model_src = np.prod(model_src, axis=1)

    else:
        policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2)))**2)/(2*variance_action))
        policy_src = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(param_policy_src[:, np.newaxis, :], state_t), axis=2)))**2)/(2*variance_action))

        model_tgt = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-(np.sum((state_t1 - state_t1_denoised_current)**2, axis=2)) / (2*variance_env[:, np.newaxis]))
        model_src = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-(np.sum((state_t1 - state_t1_denoised)**2, axis=2)) / (2*variance_env[:, np.newaxis]))

        mask = trajectories_length[:, np.newaxis] < np.repeat(np.arange(0, state_t.shape[1])[np.newaxis,:], repeats= state_t.shape[0], axis=0)
        policy_tgt[mask] = 1
        policy_src[mask] = 1
        model_tgt[mask] = 1
        model_src[mask] = 1

        policy_tgt = np.cumprod(policy_tgt, axis=1)
        policy_src = np.cumprod(policy_src, axis=1)

        model_tgt = np.cumprod(model_tgt, axis=1)
        model_src = np.cumprod(model_src, axis=1)


    weights = policy_tgt / policy_src * model_tgt / model_src

    return [weights, 0]


def computeMultipleImportanceWeightsSourceTarget(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size, compute_n_def=0):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions, trajectories_length] = getEpisodesInfoFromSource(source_dataset, env_param)
    variance_action = simulation_param.variance_action

    n = state_t.shape[0]
    param_indices_policy = np.concatenate(([0], np.cumsum(np.delete(source_dataset.episodes_per_config, -1))))
    param_indices_env = np.concatenate(([0], np.cumsum(np.delete(source_dataset.episodes_per_config[:source_dataset.n_config_cv], -1))))
    n_configuration_tgt = source_dataset.episodes_per_config[source_dataset.n_config_cv:].shape[0]

    combination_src_parameters = param_policy_src[param_indices_policy, :]#policy parameter of source not repeated
    combination_src_parameters_env = env_param_src[param_indices_env, :]#policy parameter of source not repeated

    evaluated_trajectories = source_dataset.source_distributions.shape[0]
    if algorithm_configuration.dicrete_estimation == 1 or algorithm_configuration.model_estimation == 0:
        state_t1_denoised_current = env_param.env.stepDenoisedCurrent(state_t, clipped_actions)
    else:
        state_t1_denoised_current = algorithm_configuration.model_estimator.transition(state_t, clipped_actions)

    variance_env = env_param_src[:, -1] # variance of the model transition

    if batch_size != 0:

        state_t = np.repeat(state_t[:, :, :, np.newaxis], combination_src_parameters.shape[0], axis=3) # state t
        state_t1 = np.repeat(state_t1[:, :, :, np.newaxis], combination_src_parameters.shape[0], axis=3) # state t+1
        unclipped_action_t = np.repeat(unclipped_action_t[:, :, np.newaxis], combination_src_parameters.shape[0], axis=2) # action t
        clipped_actions = np.repeat(clipped_actions[:, :, np.newaxis], combination_src_parameters.shape[0], axis=2) # clipped action t

        state_t1_denoised_src_env = env_param.env.stepDenoised(combination_src_parameters_env, state_t[:, :, :, 0:param_indices_env.shape[0]], clipped_actions[:, :, 0:param_indices_env.shape[0]])
        state_t1_denoised_current = np.repeat(state_t1_denoised_current[:, :, :, np.newaxis], n_configuration_tgt, axis=3)
        state_t1_denoised = np.concatenate([state_t1_denoised_src_env, state_t1_denoised_current], axis=3)

        mask_new_trajectories = trajectories_length[evaluated_trajectories:, np.newaxis] < np.repeat(np.arange(0, state_t.shape[1])[np.newaxis, :], repeats=state_t.shape[0]-evaluated_trajectories, axis=0)

        policy_src_new_traj = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[evaluated_trajectories:, :, :] - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, :], state_t[evaluated_trajectories:, :, :, :]), axis=2))**2)/(2*variance_action))
        model_src_new_traj = 1/np.sqrt((2*m.pi*variance_env[evaluated_trajectories:, np.newaxis, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1[evaluated_trajectories:, :, :, :] - state_t1_denoised[evaluated_trajectories:, :, :, :])**2, axis=2) / (2*variance_env[evaluated_trajectories:, np.newaxis, np.newaxis]))

        policy_src_new_traj[mask_new_trajectories] = 1
        model_src_new_traj[mask_new_trajectories] = 1

        policy_src_new_traj = np.prod(policy_src_new_traj, axis=1)
        model_src_new_traj = np.prod(model_src_new_traj, axis=1)

        if compute_n_def == 1 or algorithm_configuration.adaptive == "No":

            policy_src_new_param = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[0:evaluated_trajectories, :, 0] - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, -1], state_t[0:evaluated_trajectories, :, :, 0]), axis=2))**2)/(2*variance_action))
            model_src_new_param = 1/np.sqrt((2*m.pi*variance_env[0:evaluated_trajectories, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1[0:evaluated_trajectories, :, :, 0] - state_t1_denoised_current[0:evaluated_trajectories, :, :, 0])**2, axis=2) / (2*variance_env[0:evaluated_trajectories, np.newaxis]))

            policy_src_new_param[source_dataset.mask_weights] = 1
            model_src_new_param[source_dataset.mask_weights] = 1

            policy_src_new_param = np.prod(policy_src_new_param, axis=1)
            model_src_new_param = np.squeeze(np.asarray(np.prod(model_src_new_param, axis=1)))

            source_dataset.source_distributions = np.concatenate((source_dataset.source_distributions, np.matrix(policy_src_new_param * model_src_new_param).T), axis=1)

        source_dataset.mask_weights = np.concatenate((source_dataset.mask_weights, mask_new_trajectories), axis=0)

        source_dataset.source_distributions = np.concatenate((source_dataset.source_distributions, np.matrix(policy_src_new_traj * model_src_new_traj)), axis=0)

        policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[:, :, 0] - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t[:, :, :, 0]), axis=2))**2)/(2*variance_action))
        model_tgt = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1[:, :, :, 0] - state_t1_denoised_current[:, :, :, 0])**2, axis=2) / (2*variance_env[:, np.newaxis]))

        policy_tgt[source_dataset.mask_weights] = 1
        model_tgt[source_dataset.mask_weights] = 1

        policy_tgt = np.prod(policy_tgt, axis=1)
        model_tgt = np.prod(model_tgt, axis=1)

    else:
        policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2))**2)/(2*variance_action))
        model_tgt = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1 - state_t1_denoised_current)**2, axis=2) / (2*variance_env[:, np.newaxis]))

        policy_tgt[source_dataset.mask_weights] = 1
        model_tgt[source_dataset.mask_weights] = 1

        policy_tgt = np.prod(policy_tgt, axis=1)
        model_tgt = np.prod(model_tgt, axis=1)

    mis_denominator = np.squeeze(np.asarray(np.sum(np.multiply(source_dataset.episodes_per_config[np.newaxis, :]/n, source_dataset.source_distributions), axis=1)))

    weights = policy_tgt * model_tgt / mis_denominator

    return [weights, mis_denominator]


def computeMultipleImportanceWeightsSourceTargetPerDecision(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size, compute_n_def=0):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions, trajectories_length] = getEpisodesInfoFromSource(source_dataset, env_param)
    variance_action = simulation_param.variance_action

    n = state_t.shape[0]
    param_indices_policy = np.concatenate(([0], np.cumsum(np.delete(source_dataset.episodes_per_config, -1))))
    param_indices_env = np.concatenate(([0], np.cumsum(np.delete(source_dataset.episodes_per_config[:source_dataset.n_config_cv], -1))))
    n_configuration_tgt = source_dataset.episodes_per_config[source_dataset.n_config_cv:].shape[0]

    combination_src_parameters = param_policy_src[param_indices_policy, :]#policy parameter of source not repeated
    combination_src_parameters_env = env_param_src[param_indices_env, :]#policy parameter of source not repeated

    evaluated_trajectories = source_dataset.source_distributions.shape[0]
    if algorithm_configuration.dicrete_estimation == 1 or algorithm_configuration.model_estimation == 0:
        state_t1_denoised_current = env_param.env.stepDenoisedCurrent(state_t, clipped_actions)
    else:
        state_t1_denoised_current = algorithm_configuration.model_estimator.transition(state_t, clipped_actions)

    variance_env = env_param_src[:, -1] # variance of the model transition

    if batch_size != 0:

        state_t = np.repeat(state_t[:, :, :, np.newaxis], combination_src_parameters.shape[0], axis=3) # state t
        state_t1 = np.repeat(state_t1[:, :, :, np.newaxis], combination_src_parameters.shape[0], axis=3) # state t+1
        unclipped_action_t = np.repeat(unclipped_action_t[:, :, np.newaxis], combination_src_parameters.shape[0], axis=2) # action t
        clipped_actions = np.repeat(clipped_actions[:, :, np.newaxis], combination_src_parameters.shape[0], axis=2) # clipped action t

        state_t1_denoised_src_env = env_param.env.stepDenoised(combination_src_parameters_env, state_t[:, :, :, 0:param_indices_env.shape[0]], clipped_actions[:, :, 0:param_indices_env.shape[0]])
        state_t1_denoised_current = np.repeat(state_t1_denoised_current[:, :, :, np.newaxis], n_configuration_tgt, axis=3)
        state_t1_denoised = np.concatenate([state_t1_denoised_src_env, state_t1_denoised_current], axis=3)

        mask_new_trajectories = trajectories_length[evaluated_trajectories:, np.newaxis] < np.repeat(np.arange(0, state_t.shape[1])[np.newaxis, :], repeats=state_t.shape[0]-evaluated_trajectories, axis=0)

        policy_src_new_traj = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[evaluated_trajectories:, :, :] - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, :], state_t[evaluated_trajectories:, :, :, :]), axis=2))**2)/(2*variance_action))
        model_src_new_traj = 1/np.sqrt((2*m.pi*variance_env[evaluated_trajectories:, np.newaxis, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1[evaluated_trajectories:, :, :, :] - state_t1_denoised[evaluated_trajectories:, :, :, :])**2, axis=2) / (2*variance_env[evaluated_trajectories:, np.newaxis, np.newaxis]))

        policy_src_new_traj[mask_new_trajectories] = 1
        model_src_new_traj[mask_new_trajectories] = 1

        policy_src_new_traj = np.cumprod(policy_src_new_traj, axis=1)
        model_src_new_traj = np.cumprod(model_src_new_traj, axis=1)

        if compute_n_def == 1 or algorithm_configuration.adaptive == "No":

            policy_src_new_param = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[0:evaluated_trajectories, :, 0] - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, -1], state_t[0:evaluated_trajectories, :, :, 0]), axis=2))**2)/(2*variance_action))
            model_src_new_param = 1/np.sqrt((2*m.pi*variance_env[0:evaluated_trajectories, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1[0:evaluated_trajectories, :, :, 0] - state_t1_denoised_current[0:evaluated_trajectories, :, :, 0])**2, axis=2) / (2*variance_env[0:evaluated_trajectories, np.newaxis]))

            policy_src_new_param[source_dataset.mask_weights] = 1
            model_src_new_param[source_dataset.mask_weights] = 1

            policy_src_new_param = np.cumprod(policy_src_new_param, axis=1)
            model_src_new_param = np.squeeze(np.asarray(np.cumprod(model_src_new_param, axis=1)))

            source_dataset.source_distributions = np.concatenate((source_dataset.source_distributions, (policy_src_new_param * model_src_new_param)[:, :, np.newaxis]), axis=2)

        source_dataset.mask_weights = np.concatenate((source_dataset.mask_weights, mask_new_trajectories), axis=0)

        source_dataset.source_distributions = np.concatenate((source_dataset.source_distributions, (policy_src_new_traj * model_src_new_traj)), axis=0)

        policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[:, :, 0] - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t[:, :, :, 0]), axis=2))**2)/(2*variance_action))
        model_tgt = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1[:, :, :, 0] - state_t1_denoised_current[:, :, :, 0])**2, axis=2) / (2*variance_env[:, np.newaxis]))

        policy_tgt[source_dataset.mask_weights] = 1
        model_tgt[source_dataset.mask_weights] = 1

        policy_tgt = np.cumprod(policy_tgt, axis=1)
        model_tgt = np.cumprod(model_tgt, axis=1)
    else:

        variance_env = env_param_src[:, -1] # variance of the model transition

        policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2))**2)/(2*variance_action))
        model_tgt = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1 - state_t1_denoised_current)**2, axis=2) / (2*variance_env[:, np.newaxis]))

        policy_tgt[source_dataset.mask_weights] = 1
        model_tgt[source_dataset.mask_weights] = 1

        policy_tgt = np.cumprod(policy_tgt, axis=1)
        model_tgt = np.cumprod(model_tgt, axis=1)

    mis_denominator = np.sum(np.multiply(source_dataset.episodes_per_config[np.newaxis, np.newaxis, :]/n, source_dataset.source_distributions), axis=2)

    weights = policy_tgt * model_tgt / mis_denominator

    return [weights, mis_denominator]


def computeCv(weights, source_dataset, mis_denominator, policy_gradients, algorithm_configuration):

    if algorithm_configuration.pd == 0:
        control_variate = np.asarray((source_dataset.source_distributions / mis_denominator[:, np.newaxis]) - np.ones(source_dataset.source_distributions.shape))

        if algorithm_configuration.baseline == 1:
            baseline_covariate = np.multiply(weights[:, np.newaxis], policy_gradients)
            if algorithm_configuration.multid_approx == 0:
                control_variate = np.repeat(control_variate[:, :, np.newaxis], policy_gradients.shape[1], axis=2)
                control_variate = np.concatenate((control_variate, baseline_covariate[:, np.newaxis, :]), axis=1)
            else:
                control_variate = np.concatenate((control_variate, np.sum(baseline_covariate, axis=1)[:, np.newaxis]), axis=1)
        else:
            control_variate = np.repeat(control_variate[:, :, np.newaxis], policy_gradients.shape[1], axis=2)

    else:
        control_variate = (source_dataset.source_distributions / mis_denominator[:, :, np.newaxis]) - np.ones(source_dataset.source_distributions.shape)
        control_variate[np.isnan(control_variate)] = 0

        if algorithm_configuration.approximation == 0:
            if algorithm_configuration.baseline == 1:
                baseline_covariate = np.multiply(weights[:, :, np.newaxis], policy_gradients)
                if algorithm_configuration.multid_approx == 1:
                    control_variate = np.concatenate((control_variate, np.sum(baseline_covariate, axis=2)[:, :, np.newaxis]), axis=2)
                    control_variate = np.repeat(control_variate[:, :, :, np.newaxis], policy_gradients.shape[2], axis=3)
                else:
                    control_variate = np.repeat(control_variate[:, :, :, np.newaxis], policy_gradients.shape[2], axis=3)
                    control_variate = np.concatenate((control_variate, baseline_covariate[:, :, np.newaxis, :]), axis=2)
            else:
                control_variate = np.repeat(control_variate[:, :, :, np.newaxis], policy_gradients.shape[2], axis=3)

        else:
            control_variate = np.sum(control_variate, axis=1)
            if algorithm_configuration.baseline == 1:
                baseline_covariate = np.sum(np.multiply(weights[:, :, np.newaxis], policy_gradients), axis=1)
                if algorithm_configuration.multid_approx == 0:
                    control_variate = np.repeat(control_variate[:, :, np.newaxis], policy_gradients.shape[2], axis=2)
                    control_variate = np.concatenate((control_variate, baseline_covariate[:, np.newaxis, :]), axis=1)
                else:
                    control_variate = np.concatenate((control_variate, np.sum(baseline_covariate, axis=1)[:, np.newaxis]), axis=1)
                    control_variate = np.repeat(control_variate[:, :, np.newaxis], policy_gradients.shape[2], axis=2)
            else:
                control_variate = np.repeat(control_variate[:, :, np.newaxis], policy_gradients.shape[2], axis=2)

    return control_variate

# Compute the update of the algorithms using different estimators

def computeEssPolicySearch(policy_param, env_param, source_dataset, variance_action, algorithm_configuration, weights):
    return [0, 0]


def computeEssIs(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration):
    weights = computeImportanceWeightsSourceTarget(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, 0)[0]
    ess = np.linalg.norm(weights, 1, axis=0)**2 / np.linalg.norm(weights, 2, axis=0)**2
    if(algorithm_configuration.pd==1):
        min_index = np.argmin(ess)
        ess = ess[min_index]
    else:
        min_index = 0
    return [ess, min_index]


def computeEssSecond(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration):
#def computeEss(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration):

    trajectories_length = getEpisodesInfoFromSource(source_dataset, env_param)[-1]
    n = trajectories_length.shape[0]

    weights = algorithm_configuration.computeWeights(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, 0)[0]

    #if ess_den == 0:
        #print("problem")
    if algorithm_configuration.pd == 0:
        ess_den = np.sum(weights ** 2, axis=0)
        ess = np.sum(weights, axis=0) * n / ess_den
        min_index = 0
    else:
        ess = np.zeros(weights.shape[1])
        for t in range(weights.shape[1]):
            weights_timestep = weights[:, t]
            ess_den = np.sum(weights_timestep ** 2, axis=0)
            ess[t] = np.sum(weights_timestep, axis=0) * n / ess_den

        min_index = np.argmin(ess)
        ess = ess[min_index]

    return [ess, min_index]

def computeEss(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration):
#def computeEssSecond(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration):

    trajectories_length = getEpisodesInfoFromSource(source_dataset, env_param)[-1]
    n = trajectories_length.shape[0]

    weights = algorithm_configuration.computeWeights(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, 0)[0]

    #if ess_den == 0:
        #print("problem")
    if algorithm_configuration.pd == 0:
        variance_weights = 1/n * np.sum((weights-1)**2)
        ess = n / (1 + variance_weights)
        min_index = 0
    else:
        ess = np.zeros(weights.shape[1])
        for t in range(weights.shape[1]):
            weights_timestep = weights[:, t]
            variance_weights = 1/n * np.sum((weights_timestep-1)**2)
            ess[t] = n / (1 + variance_weights)

        min_index = np.argmin(ess)
        ess = ess[min_index]

    return [ess, min_index]


def generateEpisodesAndComputeRewards(env_param, simulation_param, param, discount_factor_timestep):

    batch_size = 5
    # Iterate for every episode in batch

    batch = createBatch(env_param.env, batch_size, env_param.episode_length, param, env_param.state_space_size, simulation_param.variance_action)[0] # [state, action, reward, next_state, next_state_unclipped, clipped_actions]

    # The return after this timestep
    total_return = np.sum(batch[:, :, env_param.state_space_size+1], axis=1)
    discounted_return = np.sum((discount_factor_timestep * batch[:, :, env_param.state_space_size+1]), axis=1)

    #Compute rewards of batch
    tot_reward_batch = np.mean(total_return)
    discounted_reward_batch = np.mean(discounted_return)

    return tot_reward_batch, discounted_reward_batch


def updateParam(env_param, source_dataset, simulation_param, param, t, m_t, v_t, algorithm_configuration, batch_size, discount_factor_timestep):

    #Generate episodes and compute rewards for the batch's statistics
    [tot_reward_batch, discounted_reward_batch] = generateEpisodesAndComputeRewards(env_param, simulation_param, param, discount_factor_timestep)


    #Compute gradients per timestep
    if algorithm_configuration.pd == 1:
        gradient_off_policy_update = np.cumsum(computeGradientsSourceTargetTimestep(param, source_dataset, simulation_param.variance_action, env_param), axis=1)

    else:
        gradient_off_policy_update = np.sum(computeGradientsSourceTargetTimestep(param, source_dataset, simulation_param.variance_action, env_param), axis=1)


    #Compute importance weights
    [weights_source_target_update, mis_denominator] = algorithm_configuration.computeWeights(param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size)

    weights_source_target_update[np.isnan(weights_source_target_update)] = 0

    if algorithm_configuration.cv == 1:
        control_variates = computeCv(weights_source_target_update, source_dataset, mis_denominator, gradient_off_policy_update, algorithm_configuration)
    else:
        control_variates = 0

    discounted_rewards_all = discount_factor_timestep * source_dataset.source_task[:, :, env_param.state_space_size+1]

    #Compute gradient for the update
    if algorithm_configuration.cv == 1:
        gradient = algorithm_configuration.computeGradientUpdate(algorithm_configuration, weights_source_target_update, gradient_off_policy_update, discounted_rewards_all, control_variates, source_dataset.n_config_cv, source_dataset, env_param)
    else:
        gradient = algorithm_configuration.computeGradientUpdate(algorithm_configuration, weights_source_target_update, gradient_off_policy_update, discounted_rewards_all, source_dataset.source_task.shape[0])

    #Update the parameter
    #param, t, m_t, v_t, gradient = alg.adam(param, -gradient, t, m_t, v_t, alpha=0.01)
    param = param + simulation_param.learning_rate * gradient

    if algorithm_configuration.off_policy == 1:
        [ess, min_index] = algorithm_configuration.computeEss(param, env_param, source_dataset, simulation_param, algorithm_configuration)

        if algorithm_configuration.pd == 1:
            ess = np.min(ess)

        if algorithm_configuration.adaptive == "Yes":
            defensive_sample = simulation_param.defensive_sample
            addEpisodesToSourceDataset(env_param, simulation_param, source_dataset, param, defensive_sample, discount_factor_timestep, simulation_param.adaptive, n_def_estimation=1)
            #Number of n_def next iteration
            num_episodes_target = computeNdef(min_index, param, env_param, source_dataset, simulation_param, algorithm_configuration)[1]
        else:
            num_episodes_target = simulation_param.batch_size

    else:
        ess = 0
        num_episodes_target = simulation_param.batch_size

    #print("Problems: n_def-" + str(num_episodes_target) + " ess-" + str(ess) + " gradient-" + str(gradient))
    print("param: " + str(param) + " tot_rewards: " + str(tot_reward_batch) + " ess: " + str(ess) + " n_def: " + str(num_episodes_target))

    return source_dataset, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, num_episodes_target

# Algorithm off policy using different estimators

def computeMultipleImportanceWeightsSourceDistributions(source_dataset, variance_action, algorithm_configuration, env_param):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions, trajectories_length] = getEpisodesInfoFromSource(source_dataset, env_param)

    param_indices = np.concatenate(([0], np.cumsum(np.delete(source_dataset.episodes_per_config, -1))))

    combination_src_parameters = (param_policy_src[param_indices, :])
    combination_src_parameters_env = (env_param_src[param_indices, :])#policy parameter of source not repeated

    state_t = np.repeat(state_t[:, :, :, np.newaxis], combination_src_parameters.shape[0], axis=3) # state t
    state_t1 = np.repeat(state_t1[:, :, :, np.newaxis], combination_src_parameters.shape[0], axis=3) # state t+1
    unclipped_action_t = np.repeat(unclipped_action_t[:, :, np.newaxis], combination_src_parameters.shape[0], axis=2) # action t
    clipped_actions = np.repeat(clipped_actions[:, :, np.newaxis], combination_src_parameters_env.shape[0], axis=2) # action t
    variance_env = env_param_src[:, -1] # variance of the model transition
    state_t1_denoised = env_param.env.stepDenoised(combination_src_parameters_env, state_t, clipped_actions)

    if algorithm_configuration.pd == 0:
        src_distributions_policy = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, :], state_t), axis=2))**2)/(2*variance_action))
        src_distributions_model = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((np.power((state_t1 - state_t1_denoised), 2)), axis=2) / (2*variance_env[:, np.newaxis, np.newaxis]))

        mask = trajectories_length[:, np.newaxis] < np.repeat(np.arange(0, state_t.shape[1])[np.newaxis, :], repeats=state_t.shape[0], axis=0)
        src_distributions_policy[mask] = 1
        src_distributions_model[mask] = 1

        src_distributions_policy = np.prod(src_distributions_policy, axis=1)
        src_distributions_model = np.prod(src_distributions_model, axis=1)

        source_dataset.mask_weights = mask

    else:
        src_distributions_policy = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, :], state_t), axis=2))**2)/(2*variance_action))
        src_distributions_model = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum(((state_t1 - state_t1_denoised)**2), axis=2) / (2*variance_env[:, np.newaxis, np.newaxis]))

        mask = trajectories_length[:, np.newaxis] < np.repeat(np.arange(0, state_t.shape[1])[np.newaxis, :], repeats=state_t.shape[0], axis=0)
        src_distributions_policy[mask] = 1
        src_distributions_model[mask] = 1

        src_distributions_policy = np.cumprod(src_distributions_policy, axis=1)
        src_distributions_model = np.cumprod(src_distributions_model, axis=1)

        source_dataset.mask_weights = mask

    return src_distributions_model * src_distributions_policy


def importanceSampling(estimator, adaptive):
    cv = 0
    pd = 0
    baseline = 0
    approximation = 0
    computeWeights = computeImportanceWeightsSourceTarget
    ess = computeEssIs
    computeGradientUpdate = onlyGradient

    algorithm_configuration = AlgorithmConfiguration(estimator, cv, pd, baseline, approximation, adaptive, computeWeights, computeGradientUpdate, ess)

    return algorithm_configuration


def pdImportanceSampling(estimator, adaptive):
    cv = 0
    pd = 1
    baseline = 0
    approximation = 0
    computeWeights = computeImportanceWeightsSourceTarget
    ess = computeEssIs
    computeGradientUpdate = onlyGradient

    algorithm_configuration = AlgorithmConfiguration(estimator, cv, pd, baseline, approximation, adaptive, computeWeights, computeGradientUpdate, ess)

    return algorithm_configuration


def multipleImportanceSampling(estimator, adaptive):
    cv = 0
    pd = 0
    baseline = 0
    approximation = 0
    computeWeights = computeMultipleImportanceWeightsSourceTarget
    ess = computeEss
    computeGradientUpdate = onlyGradient

    algorithm_configuration = AlgorithmConfiguration(estimator, cv, pd, baseline, approximation, adaptive, computeWeights, computeGradientUpdate, ess)

    return algorithm_configuration


def multipleImportanceSamplingCv(estimator, adaptive):
    cv = 1
    pd = 0
    baseline = 0
    approximation = 0
    computeWeights = computeMultipleImportanceWeightsSourceTarget
    ess = computeEss
    computeGradientUpdate = gradientAndRegression

    algorithm_configuration = AlgorithmConfiguration(estimator, cv, pd, baseline, approximation, adaptive, computeWeights, computeGradientUpdate, ess)

    return algorithm_configuration


def multipleImportanceSamplingCvBaseline(estimator, adaptive):
    cv = 1
    pd = 0
    baseline = 1
    approximation = 0
    computeWeights = computeMultipleImportanceWeightsSourceTarget
    ess = computeEss
    computeGradientUpdate = gradientAndRegression

    algorithm_configuration = AlgorithmConfiguration(estimator, cv, pd, baseline, approximation, adaptive, computeWeights, computeGradientUpdate, ess)

    return algorithm_configuration


def pdMultipleImportanceSampling(estimator, adaptive):
    cv = 0
    pd = 1
    baseline = 0
    approximation = 0
    computeWeights = computeMultipleImportanceWeightsSourceTargetPerDecision
    ess = computeEss
    computeGradientUpdate = onlyGradient

    algorithm_configuration = AlgorithmConfiguration(estimator, cv, pd, baseline, approximation, adaptive, computeWeights, computeGradientUpdate, ess)

    return algorithm_configuration


def pdMultipleImportanceSamplingCv(estimator, adaptive):
    cv = 1
    pd = 1
    baseline = 0
    approximation = 1
    computeWeights = computeMultipleImportanceWeightsSourceTargetPerDecision
    ess = computeEss
    computeGradientUpdate = gradientAndRegression

    algorithm_configuration = AlgorithmConfiguration(estimator, cv, pd, baseline, approximation, adaptive, computeWeights, computeGradientUpdate, ess)

    return algorithm_configuration


def pdMultipleImportanceSamplingCvBaselineApprox(estimator, adaptive):
    cv = 1
    pd = 1
    baseline = 1
    approximation = 1
    computeWeights = computeMultipleImportanceWeightsSourceTargetPerDecision
    ess = computeEss
    computeGradientUpdate = gradientAndRegression

    algorithm_configuration = AlgorithmConfiguration(estimator, cv, pd, baseline, approximation, adaptive, computeWeights, computeGradientUpdate, ess)

    return algorithm_configuration


def pdMultipleImportanceSamplingCvBaseline(estimator, adaptive):
    cv = 1
    pd = 1
    baseline = 1
    approximation = 0
    computeWeights = computeMultipleImportanceWeightsSourceTargetPerDecision
    ess = computeEss
    computeGradientUpdate = gradientAndRegression

    algorithm_configuration = AlgorithmConfiguration(estimator, cv, pd, baseline, approximation, adaptive, computeWeights, computeGradientUpdate, ess)

    return algorithm_configuration


def reinforce(estimator, adaptive):
    cv = 0
    pd = 0
    baseline = 0
    approximation = 0
    adaptive_alg = "No"
    computeWeights = weightsPolicySearch
    ess = computeEssPolicySearch
    computeGradientUpdate = gradientPolicySearch

    algorithm_configuration = AlgorithmConfiguration(estimator, cv, pd, baseline, approximation, adaptive_alg, computeWeights, computeGradientUpdate, ess)

    return algorithm_configuration


def reinforceBaseline(estimator, adaptive):
    cv = 0
    pd = 0
    baseline = 1
    approximation = 0
    adaptive_alg = "No"
    computeWeights = weightsPolicySearch
    ess = computeEssPolicySearch
    computeGradientUpdate = gradientPolicySearch

    algorithm_configuration = AlgorithmConfiguration(estimator, cv, pd, baseline, approximation, adaptive_alg, computeWeights, computeGradientUpdate, ess)

    return algorithm_configuration


def gpomdp(estimator, adaptive):
    cv = 0
    pd = 1
    baseline = 1
    approximation = 0
    adaptive_alg = "No"
    computeWeights = weightsPolicySearch
    ess = computeEssPolicySearch
    computeGradientUpdate = gradientPolicySearch

    algorithm_configuration = AlgorithmConfiguration(estimator, cv, pd, baseline, approximation, adaptive_alg, computeWeights, computeGradientUpdate, ess)

    return algorithm_configuration


def switch_estimator(estimator, adaptive):

    if estimator == "IS":
        algorithm_configuration = importanceSampling(estimator, adaptive)
    if estimator == "PD-IS":
        algorithm_configuration = pdImportanceSampling(estimator, adaptive)
    if estimator == "MIS":
        algorithm_configuration = multipleImportanceSampling(estimator, adaptive)
    if estimator == "MIS-CV":
        algorithm_configuration = multipleImportanceSamplingCv(estimator, adaptive)
    if estimator == "MIS-CV-BASELINE":
        algorithm_configuration = multipleImportanceSamplingCvBaseline(estimator, adaptive)
    if estimator == "PD-MIS":
        algorithm_configuration = pdMultipleImportanceSampling(estimator, adaptive)
    if estimator == "PD-MIS-CV":
        algorithm_configuration = pdMultipleImportanceSamplingCv(estimator, adaptive)
    if estimator == "PD-MIS-CV-BASELINE-APPROXIMATED":
        algorithm_configuration = pdMultipleImportanceSamplingCvBaselineApprox(estimator, adaptive)
    if estimator == "PD-MIS-CV-BASELINE":
        algorithm_configuration = pdMultipleImportanceSamplingCvBaseline(estimator, adaptive)
    if estimator == "REINFORCE":
        algorithm_configuration = reinforce(estimator, adaptive)
    if estimator == "REINFORCE-BASELINE":
        algorithm_configuration = reinforceBaseline(estimator, adaptive)
    if estimator == "GPOMDP":
        algorithm_configuration = gpomdp(estimator, adaptive)

    return algorithm_configuration


def addEpisodesToSourceDataset(env_param, simulation_param, source_dataset, param, num_episodes_target, discount_factor_timestep, adaptive, n_def_estimation=0):

    batch_size = num_episodes_target

    source_param_new = np.zeros((batch_size, 1+env_param.param_space_size+env_param.env_param_space_size+1))
    source_task_new = np.zeros((batch_size, env_param.episode_length, env_param.state_space_size + 2 + env_param.state_space_size))
    # Iterate for every episode in batch

    [batch, trajectory_length] = createBatch(env_param.env, batch_size, env_param.episode_length, param, env_param.state_space_size, simulation_param.variance_action) # [state, action, reward, next_state, next_state_unclipped, clipped_actions]

    source_task_new[:, :, 0:env_param.state_space_size] = batch[:, :, 0:env_param.state_space_size] # state
    source_task_new[:, :, env_param.state_space_size] = batch[:, :, env_param.state_space_size+2+env_param.state_space_size+env_param.state_space_size] # unclipped action
    source_task_new[:, :, env_param.state_space_size+1] = batch[:, :, env_param.state_space_size+1] # reward
    source_task_new[:, :, env_param.state_space_size+2:env_param.state_space_size+2+env_param.state_space_size] = batch[:, :, env_param.state_space_size+2:env_param.state_space_size+2+env_param.state_space_size] #next state

    #unclipped next_states and actions
    next_states_unclipped_new = batch[:, :, env_param.state_space_size+2+env_param.state_space_size:env_param.state_space_size+2+env_param.state_space_size+env_param.state_space_size]
    next_states_unclipped_denoised_new = batch[:, :, env_param.state_space_size+2+env_param.state_space_size+env_param.state_space_size+1:env_param.state_space_size+2+env_param.state_space_size+env_param.state_space_size+1+env_param.state_space_size]
    clipped_actions_new = batch[:, :, env_param.state_space_size]

    # The return after this timestep
    discounted_return_timestep = (discount_factor_timestep * batch[:, :, env_param.state_space_size+1])
    discounted_return = np.sum(discounted_return_timestep, axis=1)

    source_param_new[:, 0] = discounted_return
    source_param_new[:, 1:1+env_param.param_space_size] = param
    source_param_new[:, 1+env_param.param_space_size:1+env_param.param_space_size+env_param.env_param_space_size] = env_param.env.getEnvParam().T
    source_param_new[:, 1+env_param.param_space_size+env_param.env_param_space_size] = trajectory_length

    # Concatenate new episodes to source tasks
    source_dataset.source_param = np.concatenate((source_dataset.source_param, source_param_new), axis=0)
    source_dataset.source_task = np.concatenate((source_dataset.source_task, source_task_new), axis=0)

    #add number of episodes per configuration when I collect episodes for the estimation
    if adaptive == "Yes":
        if n_def_estimation == 0:
            source_dataset.episodes_per_config[-1] = source_dataset.episodes_per_config[-1] + batch_size
        if n_def_estimation == 1:
            source_dataset.episodes_per_config = np.concatenate((source_dataset.episodes_per_config, [batch_size]))
    else:
        source_dataset.episodes_per_config = np.concatenate((source_dataset.episodes_per_config, [batch_size]))

    source_dataset.next_states_unclipped = np.concatenate((source_dataset.next_states_unclipped, next_states_unclipped_new), axis=0)
    source_dataset.next_states_unclipped_denoised = np.concatenate((source_dataset.next_states_unclipped_denoised, next_states_unclipped_denoised_new), axis=0)
    source_dataset.clipped_actions = np.concatenate((source_dataset.clipped_actions, clipped_actions_new), axis=0)

    return [source_task_new, source_param_new, batch_size, next_states_unclipped_new, clipped_actions_new, next_states_unclipped_denoised_new]


def setEnvParametersTarget(env, source_dataset, env_param):

    source_length = source_dataset.initial_size
    source_dataset.source_param[source_length:, 1+env_param.param_space_size:1+env_param.param_space_size+env_param.env_param_space_size] = env.getEnvParam().T


def learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1, model_estimation=0, dicrete_estimation=1, multid_approx=0, model_estimator=None, verbose=True):

    param = np.random.normal(simulation_param.mean_initial_param, simulation_param.variance_initial_param)

    discount_factor_timestep = np.power(simulation_param.discount_factor*np.ones(env_param.episode_length), range(env_param.episode_length))
    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Keep track of useful statistics#
    stats = BatchStats(simulation_param.num_batch, env_param.param_space_size)
    n_def = simulation_param.batch_size
    ess = 0

    algorithm_configuration = switch_estimator(estimator, simulation_param.adaptive)
    algorithm_configuration.off_policy = off_policy
    algorithm_configuration.multid_approx = multid_approx
    algorithm_configuration.dicrete_estimation = dicrete_estimation
    algorithm_configuration.model_estimation = model_estimation
    algorithm_configuration.model_estimator = model_estimator

    if off_policy == 1:
        if re.match("^.*MIS.*", estimator):
            source_dataset.source_distributions = computeMultipleImportanceWeightsSourceDistributions(source_dataset, simulation_param.variance_action, algorithm_configuration, env_param)
            [ess, min_index] = algorithm_configuration.computeEss(param, env_param, source_dataset, simulation_param, algorithm_configuration)

        else:
            [ess, min_index] = algorithm_configuration.computeEss(param, env_param, source_dataset, simulation_param, algorithm_configuration)

        if simulation_param.adaptive == "Yes":
            defensive_sample = simulation_param.defensive_sample
            addEpisodesToSourceDataset(env_param, simulation_param, source_dataset, param, defensive_sample, discount_factor_timestep, simulation_param.adaptive, n_def_estimation=1)
            n_def = computeNdef(min_index, param, env_param, source_dataset, simulation_param, algorithm_configuration)[1]

        else:
            if simulation_param.adaptive == "Yes":
                n_def = simulation_param.ess_min

    for i_batch in range(simulation_param.num_batch):

        if verbose:
            print("Iteration {0}".format(i_batch))
            print(estimator)

        batch_size = n_def

        if simulation_param.adaptive == "Yes":
            n_def = n_def + simulation_param.defensive_sample

        stats.n_def[i_batch] = n_def
        stats.ess[i_batch] = ess

        if batch_size != 0:
            #Generate the episodes and compute the rewards over the batch
            if verbose:
                print("Collecting {0} episodes...".format(batch_size))
                start = time.time()
            [source_task_tgt, source_param_tgt, episodes_per_configuration_tgt, next_states_unclipped_tgt, actions_clipped_tgt, next_states_unclipped_denoised_tgt] = addEpisodesToSourceDataset(env_param, simulation_param, source_dataset, param, batch_size, discount_factor_timestep, algorithm_configuration.adaptive, n_def_estimation=0)
            dataset_model_estimation = sc.SourceDataset(source_task_tgt, source_param_tgt, episodes_per_configuration_tgt, next_states_unclipped_tgt, actions_clipped_tgt, next_states_unclipped_denoised_tgt, 1)
            if verbose:
                print("Done collecting episodes ({0}s)".format(time.time()-start))

        if model_estimation == 1:
            if verbose:
                print("Updating model...")
                start = time.time()

            if dicrete_estimation == 1:
                env = model_estimator.chooseTransitionModel(env_param, param, simulation_param, source_dataset.source_param, source_dataset.episodes_per_config, source_dataset.n_config_cv, source_dataset.initial_size, dataset_model_estimation)
                setEnvParametersTarget(env, source_dataset, env_param)
            else:
                model_estimator.update_model(source_dataset)

            if verbose:
                print("Done updating model ({0}s)".format(time.time() - start))

        if verbose:
            print("Updating policy...")
            start = time.time()

        [source_dataset, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, n_def] = updateParam(env_param, source_dataset, simulation_param, param, t, m_t, v_t, algorithm_configuration, batch_size, discount_factor_timestep)

        if verbose:
            print("Done updating policy ({0}s)".format(time.time() - start))

        # Update statistics
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        stats.policy_parameter[i_batch, :] = param
        stats.gradient[i_batch, :] = gradient
    return stats
