import math as m
import numpy as np
from collections import namedtuple
import algorithmPolicySearch as alg
import random
import re

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


def createBatch(env, batch_size, episode_length, param, state_space_size, variance_action):

    information_size = state_space_size+2+state_space_size+state_space_size+1+state_space_size
    batch = np.zeros((batch_size, episode_length, information_size)) #[state, clipped_action, reward, next_state, unclipped_state, action]
    for i_batch in range(batch_size):
        state = env.reset()

        for t in range(episode_length):
            # Take a step
            mean_action = np.sum(np.multiply(param, state))
            action = np.random.normal(mean_action, m.sqrt(variance_action))
            next_state, reward, done, unclipped_state, clipped_action, next_state_denoised = env.step(action)
            # Keep track of the transition
            #env.render()
            batch[i_batch, t, 0:state_space_size] = state
            batch[i_batch, t, state_space_size] = clipped_action
            batch[i_batch, t, state_space_size+1] = reward
            batch[i_batch, t, state_space_size+2:state_space_size+2+state_space_size] = next_state
            batch[i_batch, t, state_space_size+2+state_space_size:state_space_size+2+state_space_size+state_space_size] = unclipped_state
            batch[i_batch, t, state_space_size+2+state_space_size+state_space_size:state_space_size+2+state_space_size+state_space_size+1] = action
            batch[i_batch, t, state_space_size+2+state_space_size+state_space_size+1:state_space_size+2+state_space_size+state_space_size+1+state_space_size] = next_state_denoised

            if done:
                break

            state = next_state

    return batch


def computeGradientsSourceTargetTimestep(param, source_dataset, variance_action, env_param):

    state_t = source_dataset.source_task[:, :, 0:env_param.state_space_size] # state t
    action_t = source_dataset.source_task[:, :, env_param.state_space_size]

    gradient_off_policy = (action_t - np.sum(np.multiply(param[np.newaxis, np.newaxis, :],  state_t), axis=2))[:, :, np.newaxis] * state_t / variance_action

    return gradient_off_policy


def computeNdef(min_index, param, env_param, source_dataset, simulation_param, algorithm_configuration):

    weights = algorithm_configuration.computeWeights(param, env_param, source_dataset, simulation_param, algorithm_configuration, 0)[0]
    n = source_dataset.source_task.shape[0]
    if algorithm_configuration.pd == 1:
        weights = weights[:, min_index]
    w_1 = np.linalg.norm(weights, 1)
    w_2 = np.linalg.norm(weights, 2)
    num_episodes_target = int(max(0, np.ceil((simulation_param.ess_min * w_1 / n) - (w_1 * n / (w_2 ** 2)))))

    return num_episodes_target

# Compute gradients

def onlyGradient(algorithm_configuration, weights_source_target_update, gradient_off_policy_update, discounted_rewards_all, N):
    if(algorithm_configuration.pd == 0):
        gradient = 1/N * np.sum((np.squeeze(np.array(weights_source_target_update))[:, np.newaxis] * gradient_off_policy_update) * np.sum(discounted_rewards_all, axis=1)[:, np.newaxis], axis=0)
    else:
        gradient = 1/N * np.sum(np.sum((np.squeeze(np.array(weights_source_target_update))[:, :, np.newaxis] * gradient_off_policy_update) * discounted_rewards_all[:, :, np.newaxis], axis=1), axis=0)

    return gradient


def regressionFitting(y, x, n_config_cv, baseline_flag):
    n_config_cv = min(n_config_cv, 100)
    if baseline_flag == 1:
        baseline = np.squeeze(np.asarray(x[:, -1]))[:, np.newaxis]
        x = np.concatenate([x[:, 0:n_config_cv], baseline], axis=1)
        #x = np.concatenate([x, baseline], axis=1)
    else:
        x = x[:, 0:n_config_cv]

    train_size = int(np.ceil(x.shape[0]/6*5))
    train_index = random.sample(range(x.shape[0]), train_size)
    x_train = x[train_index]
    y_train = y[train_index]
    x_test = np.delete(x, train_index, axis=0)
    y_test = np.delete(y, train_index, axis=0)
    beta = np.squeeze(np.asarray(np.matmul(np.linalg.pinv(x_train[:, 1:]), y_train)))
    error = y_test - np.dot(x_test[:, 1:], beta)

    # x_avg = np.squeeze(np.asarray(np.mean(x, axis=0)))
    # beta = np.matmul(np.linalg.inv(np.matmul((x[:, 1:]-x_avg[1:]).T, (x[:, 1:]-x_avg[1:]))), np.matmul((x[:, 1:]-x_avg[1:]).T, (y-y_avg)).T)
    # I = np.identity(y.shape[0])
    # error = np.squeeze(np.asarray(np.matmul(I - np.matmul(x[:, 1:], np.linalg.pinv(x[:, 1:])), y)))

    return np.mean(error)


def gradientAndRegression(algorithm_configuration, weights_source_target_update, gradient_off_policy_update, discounted_rewards_all, control_variates, n_config_cv):

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
        for i in range(gradient_estimation.shape[2]):
        # Fitting the regression for every t 0...T-1
            for t in range(control_variates.shape[1]):
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

    return [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions]


def weightsPolicySearch(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size):


    if algorithm_configuration.pd == 0:
        weights = np.zeros(source_dataset.source_task.shape[0]-simulation_param.batch_size)
        weights = np.concatenate((weights, np.ones(simulation_param.batch_size)), axis=0)
    else:
        weights = np.zeros((source_dataset.source_task.shape[0]-simulation_param.batch_size, env_param.episode_length))
        weights = np.concatenate((weights, np.ones((simulation_param.batch_size, env_param.episode_length))), axis=0)

    return [weights, 0]

def computeImportanceWeightsSourceTarget(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_action_t] = getEpisodesInfoFromSource(source_dataset, env_param)
    variance_action = simulation_param.variance_action
    variance_env = env_param_src[:, -1]
    param_indices = np.concatenate(([0], np.cumsum(np.delete(source_dataset.episodes_per_config, -1))))
    state_t1_denoised_current = env_param.env.stepDenoisedCurrent(state_t, clipped_action_t) #change to source dataset . states denoised
    state_t1_denoised = source_dataset.next_states_unclipped_denoised

    if algorithm_configuration.pd == 0:
        policy_tgt = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2)))**2)/(2*variance_action)), axis=1)
        policy_src = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(param_policy_src[:, np.newaxis, :], state_t), axis=2)))**2)/(2*variance_action)), axis=1)

        model_tgt = np.prod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis]) * np.exp(-(np.sum((state_t1 - state_t1_denoised_current)**2, axis=2)) / (2*variance_env[:, np.newaxis])), axis=1)
        model_src = np.prod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis]) * np.exp(-(np.sum((state_t1 - state_t1_denoised)**2, axis=2)) / (2*variance_env[:, np.newaxis])), axis=1)

    else:
        policy_tgt = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2)))**2)/(2*variance_action)), axis=1)
        policy_src = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(param_policy_src[:, np.newaxis, :], state_t), axis=2)))**2)/(2*variance_action)), axis=1)

        model_tgt = np.cumprod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis]) * np.exp(-(np.sum((state_t1 - state_t1_denoised_current)**2, axis=2)) / (2*variance_env[:, np.newaxis])), axis=1)
        model_src = np.cumprod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis]) * np.exp(-(np.sum((state_t1 - state_t1_denoised)**2, axis=2)) / (2*variance_env[:, np.newaxis])), axis=1)

    weights = policy_tgt / policy_src * model_tgt / model_src

    return [weights, 0]


def computeMultipleImportanceWeightsSourceTarget(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions] = getEpisodesInfoFromSource(source_dataset, env_param)
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

        policy_src_new_traj = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[evaluated_trajectories:, :, :] - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, :], state_t[evaluated_trajectories:, :, :, :]), axis=2))**2)/(2*variance_action)), axis=1)
        model_src_new_traj = np.prod(1/np.sqrt(2*m.pi*variance_env[evaluated_trajectories:, np.newaxis, np.newaxis]) * np.exp(-np.sum((state_t1[evaluated_trajectories:, :, :] - state_t1_denoised[evaluated_trajectories:, :, :])**2, axis=2) / (2*variance_env[evaluated_trajectories:, np.newaxis, np.newaxis])), axis=1)

        policy_src_new_param = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[0:evaluated_trajectories, :, 0] - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, -1], state_t[0:evaluated_trajectories, :, :, 0]), axis=2))**2)/(2*variance_action)), axis=1)
        model_src_new_param = np.prod(1/np.sqrt(2*m.pi*variance_env[0:evaluated_trajectories, np.newaxis]) * np.exp(-np.sum((state_t1[0:evaluated_trajectories, :, 0] - state_t1_denoised[0:evaluated_trajectories, :, 0])**2, axis=2) / (2*variance_env[0:evaluated_trajectories, np.newaxis])), axis=1)

        source_dataset.source_distributions = np.concatenate((source_dataset.source_distributions, np.matrix(policy_src_new_param * model_src_new_param).T), axis=1)
        source_dataset.source_distributions = np.concatenate((source_dataset.source_distributions, np.matrix(policy_src_new_traj * model_src_new_traj)), axis=0)

        policy_tgt = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[:, :, 0] - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t[:, :, :, 0]), axis=2))**2)/(2*variance_action)), axis=1)
        model_tgt = np.prod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis]) * np.exp(-np.sum((state_t1[:, :, :, 0] - state_t1_denoised_current)**2, axis=2) / (2*variance_env[:, np.newaxis])), axis=1)

    else:

        variance_env = env_param_src[:, -1] # variance of the model transition

        policy_tgt = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2))**2)/(2*variance_action)), axis=1)
        model_tgt = np.prod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis]) * np.exp(-np.sum((state_t1 - state_t1_denoised_current)**2, axis=2) / (2*variance_env[:, np.newaxis])), axis=1)

    mis_denominator = np.squeeze(np.asarray(np.sum(np.multiply(source_dataset.episodes_per_config[np.newaxis, :]/n, source_dataset.source_distributions), axis=1)))

    weights = policy_tgt * model_tgt / mis_denominator

    return [weights, mis_denominator]


def computeMultipleImportanceWeightsSourceTargetPerDecision(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions] = getEpisodesInfoFromSource(source_dataset, env_param)
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
        model_src_new_traj = np.cumprod(1/np.sqrt(2*m.pi*variance_env[evaluated_trajectories:, np.newaxis, np.newaxis]) * np.exp(-np.sum((state_t1[evaluated_trajectories:, :, :] - state_t1_denoised[evaluated_trajectories:, :, :])**2, axis=2) / (2*variance_env[evaluated_trajectories:, np.newaxis, np.newaxis])), axis=1)

        policy_src_new_param = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[0:evaluated_trajectories, :, 0] - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, -1], state_t[0:evaluated_trajectories, :, :, 0]), axis=2))**2)/(2*variance_action)), axis=1)
        model_src_new_param = np.cumprod(1/np.sqrt(2*m.pi*variance_env[0:evaluated_trajectories, np.newaxis]) * np.exp(-np.sum((state_t1[0:evaluated_trajectories, :, 0] - state_t1_denoised[0:evaluated_trajectories, :, 0])**2, axis=2) / (2*variance_env[0:evaluated_trajectories, np.newaxis])), axis=1)

        source_dataset.source_distributions = np.concatenate((source_dataset.source_distributions, (policy_src_new_param * model_src_new_param)[:, :, np.newaxis]), axis=2)
        source_dataset.source_distributions = np.concatenate((source_dataset.source_distributions, (policy_src_new_traj * model_src_new_traj)), axis=0)

        policy_tgt = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t[:, :, 0] - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t[:, :, :, 0]), axis=2))**2)/(2*variance_action)), axis=1)
        model_tgt = np.cumprod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis]) * np.exp(-np.sum((state_t1[:, :, :, 0] - state_t1_denoised_current)**2, axis=2) / (2*variance_env[:, np.newaxis])), axis=1)

    else:

        variance_env = env_param_src[:, -1] # variance of the model transition

        policy_tgt = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2))**2)/(2*variance_action)), axis=1)
        model_tgt = np.cumprod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis]) * np.exp(-np.sum((state_t1 - state_t1_denoised_current)**2, axis=2) / (2*variance_env[:, np.newaxis])), axis=1)

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


def computeEss(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions] = getEpisodesInfoFromSource(source_dataset, env_param)
    variance_env = env_param_src[:, -1]
    state_t1_denoised_current = env_param.env.stepDenoisedCurrent(state_t, clipped_actions)
    variance_action = simulation_param.variance_action
    n = state_t1_denoised_current.shape[0]

    if algorithm_configuration.pd == 0:
        policy_tgt = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2)))**2)/(2*variance_action)), axis=1)
        model_tgt = np.prod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis]) * np.exp(-(np.sum((state_t1 - state_t1_denoised_current)**2, axis=2)) / (2*variance_env[:, np.newaxis])), axis=1)
        mis_denominator = np.squeeze(np.asarray(np.sum(np.multiply(source_dataset.episodes_per_config[np.newaxis, :], source_dataset.source_distributions), axis=1)))

    else:
        policy_tgt = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2)))**2)/(2*variance_action)), axis=1)
        model_tgt = np.cumprod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis]) * np.exp(-(np.sum((state_t1 - state_t1_denoised_current)**2, axis=2)) / (2*variance_env[:, np.newaxis])), axis=1)
        mis_denominator = np.sum(np.multiply(source_dataset.episodes_per_config[np.newaxis, np.newaxis, :], source_dataset.source_distributions), axis=2)

    numerator = np.power(policy_tgt * model_tgt, 2)
    denominator = np.power(mis_denominator, 2)
    ess_inv = numerator/denominator
    if np.sum(ess_inv) == 0:
        print("problem")
    ess = 1/np.sum(ess_inv, axis=0)

    if(algorithm_configuration.pd==1):
        min_index = np.argmin(ess)
        ess = ess[min_index]
    else:
        min_index = 0

    return [ess, min_index]


def addEpisodesToSourceDataset(env_param, simulation_param, source_dataset, param, num_episodes_target, discount_factor_timestep):

    batch_size = num_episodes_target

    source_param_new = np.zeros((batch_size, 1+env_param.param_space_size+env_param.env_param_space_size))
    source_task_new = np.zeros((batch_size, env_param.episode_length, env_param.state_space_size + 2 + env_param.state_space_size))
    # Iterate for every episode in batch

    batch = createBatch(env_param.env, batch_size, env_param.episode_length, param, env_param.state_space_size, simulation_param.variance_action) # [state, action, reward, next_state, next_state_unclipped, clipped_actions]

    source_task_new[:, :, 0:env_param.state_space_size] = batch[:, :, 0:env_param.state_space_size] # state
    source_task_new[:, :, env_param.state_space_size] = batch[:, :, env_param.state_space_size+2+env_param.state_space_size+env_param.state_space_size] # unclipped action
    source_task_new[:, :, env_param.state_space_size+1] = batch[:, :, env_param.state_space_size+1] # reward
    source_task_new[:, :, env_param.state_space_size+2:env_param.state_space_size+2+env_param.state_space_size] = batch[:, :, env_param.state_space_size+2:env_param.state_space_size+2+env_param.state_space_size] #next state

    #unclipped next_states and actions
    next_states_unclipped_new = batch[:, :, env_param.state_space_size+2+env_param.state_space_size:env_param.state_space_size+2+env_param.state_space_size+env_param.state_space_size]
    next_states_unclipped_denoised_new = batch[:, :, env_param.state_space_size+2+env_param.state_space_size+env_param.state_space_size+1:env_param.state_space_size+2+env_param.state_space_size+env_param.state_space_size+1+env_param.state_space_size]
    clipped_actions_new = batch[:, :, env_param.state_space_size]

    # The return after this timestep
    total_return = np.sum(batch[:, :, env_param.state_space_size+1], axis=1)
    discounted_return_timestep = (discount_factor_timestep * batch[:, :, env_param.state_space_size+1])
    discounted_return = np.sum(discounted_return_timestep, axis=1)

    source_param_new[:, 0] = discounted_return
    source_param_new[:, 1:1+env_param.param_space_size] = param
    source_param_new[:, 1+env_param.param_space_size:1+env_param.param_space_size+env_param.env_param_space_size] = env_param.env.getEnvParam().T

    # Concatenate new episodes to source tasks
    source_dataset.source_param = np.concatenate((source_dataset.source_param, source_param_new), axis=0)
    source_dataset.source_task = np.concatenate((source_dataset.source_task, source_task_new), axis=0)
    source_dataset.episodes_per_config = np.concatenate((source_dataset.episodes_per_config, [batch_size]))
    source_dataset.next_states_unclipped = np.concatenate((source_dataset.next_states_unclipped, next_states_unclipped_new), axis=0)
    source_dataset.next_states_unclipped_denoised = np.concatenate((source_dataset.next_states_unclipped_denoised, next_states_unclipped_denoised_new), axis=0)
    source_dataset.clipped_actions = np.concatenate((source_dataset.clipped_actions, clipped_actions_new), axis=0)


def generateEpisodesAndComputeRewards(env_param, simulation_param, param, discount_factor_timestep):

    batch_size = 5
    # Iterate for every episode in batch

    batch = createBatch(env_param.env, batch_size, env_param.episode_length, param, env_param.state_space_size, simulation_param.variance_action) # [state, action, reward, next_state, next_state_unclipped, clipped_actions]

    # The return after this timestep
    total_return = np.sum(batch[:, :, env_param.state_space_size+1], axis=1)
    discounted_return = np.sum((discount_factor_timestep * batch[:, :, env_param.state_space_size+1]), axis=1)

    #Compute rewards of batch
    tot_reward_batch = np.mean(total_return)
    discounted_reward_batch = np.mean(discounted_return)

    return tot_reward_batch, discounted_reward_batch


def updateParam(env_param, source_dataset, simulation_param, param, t, m_t, v_t, algorithm_configuration, batch_size):

    num_episodes_target = batch_size
    discount_factor_timestep = np.power(simulation_param.discount_factor*np.ones(env_param.episode_length), range(env_param.episode_length))

    #Collect new episodes
    if num_episodes_target != 0:
        #Generate the episodes and compute the rewards over the batch
        addEpisodesToSourceDataset(env_param, simulation_param, source_dataset, param, num_episodes_target, discount_factor_timestep)

    #Generate episodes and compute rewards
    [tot_reward_batch, discounted_reward_batch] = generateEpisodesAndComputeRewards(env_param, simulation_param, param, discount_factor_timestep)


    #Compute gradients per timestep
    if algorithm_configuration.pd == 1:
        gradient_off_policy_update = np.cumsum(computeGradientsSourceTargetTimestep(param, source_dataset, simulation_param.variance_action, env_param), axis=1)

    else:
        gradient_off_policy_update = np.sum(computeGradientsSourceTargetTimestep(param, source_dataset, simulation_param.variance_action, env_param), axis=1)


    #Compute importance weights
    [weights_source_target_update, mis_denominator] = algorithm_configuration.computeWeights(param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size)

    if algorithm_configuration.cv == 1:
        control_variates = computeCv(weights_source_target_update, source_dataset, mis_denominator, gradient_off_policy_update, algorithm_configuration)
    else:
        control_variates = 0

    weights_source_target_update[np.isnan(weights_source_target_update)] = 0
    discounted_rewards_all = discount_factor_timestep * source_dataset.source_task[:, :, env_param.state_space_size+1]

    #Compute gradient for the update
    if algorithm_configuration.cv == 1:
        gradient = algorithm_configuration.computeGradientUpdate(algorithm_configuration, weights_source_target_update, gradient_off_policy_update, discounted_rewards_all, control_variates, source_dataset.n_config_cv)
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
            #Number of n_def next iteration
            num_episodes_target = computeNdef(min_index, param, env_param, source_dataset, simulation_param, algorithm_configuration)
        else:
            num_episodes_target = simulation_param.batch_size

    else:
        ess = 0
        num_episodes_target = simulation_param.batch_size

    #print("Problems: n_def-" + str(num_episodes_target) + " ess-" + str(ess) + " gradient-" + str(gradient))
    print("param: " + str(param) + " gradient: " + str(gradient) + " ess: " + str(ess))

    return source_dataset, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, num_episodes_target

# Algorithm off policy using different estimators

def computeMultipleImportanceWeightsSourceDistributions(source_dataset, variance_action, algorithm_configuration, env_param):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions] = getEpisodesInfoFromSource(source_dataset, env_param)

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
        src_distributions_policy = np.prod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, :], state_t), axis=2))**2)/(2*variance_action)), axis=1)
        src_distributions_model = np.prod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis, np.newaxis]) * np.exp(-np.sum((np.power((state_t1 - state_t1_denoised), 2)), axis=2) / (2*variance_env[:, np.newaxis, np.newaxis])), axis=1)
    else:
        src_distributions_policy = np.cumprod(1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, :], state_t), axis=2))**2)/(2*variance_action)), axis=1)
        src_distributions_model = np.cumprod(1/np.sqrt(2*m.pi*variance_env[:, np.newaxis, np.newaxis]) * np.exp(-np.sum(((state_t1 - state_t1_denoised)**2), axis=2) / (2*variance_env[:, np.newaxis, np.newaxis])), axis=1)

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
    if estimator== "PD-IS":
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


def learnPolicy(env_param, simulation_param, source_dataset, estimator, off_policy=1, multid_approx=0):

    param = np.random.normal(simulation_param.mean_initial_param, simulation_param.variance_initial_param)

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

    if off_policy == 1:
        if re.match("^.*MIS.*", estimator):
            source_dataset.source_distributions = computeMultipleImportanceWeightsSourceDistributions(source_dataset, simulation_param.variance_action, algorithm_configuration, env_param)
            [ess, min_index] = algorithm_configuration.computeEss(param, env_param, source_dataset, simulation_param, algorithm_configuration)

        else:
            [ess, min_index] = algorithm_configuration.computeEss(param, env_param, source_dataset, simulation_param, algorithm_configuration)

        if simulation_param.adaptive == "Yes":
            n_def = computeNdef(min_index, param, env_param, source_dataset, simulation_param, algorithm_configuration)

    for i_batch in range(simulation_param.num_batch):

        stats.n_def[i_batch] = n_def
        batch_size = n_def
        stats.ess[i_batch] = ess

        [source_dataset, param, t, m_t, v_t, tot_reward_batch, discounted_reward_batch, gradient, ess, n_def] = updateParam(env_param, source_dataset, simulation_param, param, t, m_t, v_t, algorithm_configuration, batch_size)

        # Update statistics
        stats.total_rewards[i_batch] = tot_reward_batch
        stats.disc_rewards[i_batch] = discounted_reward_batch
        stats.policy_parameter[i_batch, :] = param
        stats.gradient[i_batch, :] = gradient
    return stats
