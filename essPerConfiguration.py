import numpy as np
import math as m
import sourceTaskCreation as stc
import gym
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

class AlgorithmConfiguration:

    def __init__(self, pd, computeWeights):

        self.pd = pd
        self.computeWeights = computeWeights

class EnvParam:

    def __init__(self, env, param_space_size, state_space_size, env_param_space_size, episode_length):

        self.env = env
        self.param_space_size = param_space_size
        self.state_space_size = state_space_size
        self.env_param_space_size = env_param_space_size
        self.episode_length = episode_length

class SimulationParam:

    def __init__(self, mean_initial_param, variance_initial_param, variance_action, batch_size, discount_factor, ess_min):

        self.mean_initial_param = mean_initial_param
        self.variance_initial_param = variance_initial_param
        self.variance_action = variance_action
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.ess_min = ess_min

class SourceDataset:

    def __init__(self, source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, next_states_unclipped_denoised):

        self.source_task = source_task
        self.source_param = source_param
        self.episodes_per_config = episodes_per_config
        self.next_states_unclipped = next_states_unclipped
        self.next_states_unclipped_denoised = next_states_unclipped_denoised
        self.clipped_actions = clipped_actions
        self.n_config_cv = episodes_per_config.shape[0]
        self.initial_size = source_task.shape[0]
        self.source_distributions = None
        self.mask_weights = None

def getEpisodesInfoFromSource(source_dataset, env_param):

    param_policy_src = source_dataset.source_param[:, 1:1+env_param.param_space_size] # policy parameter of source
    state_t = source_dataset.source_task[:, :, 0:env_param.state_space_size] # state t
    state_t1 = source_dataset.next_states_unclipped # state t+1
    unclipped_action_t = source_dataset.source_task[:, :, env_param.state_space_size]
    clipped_actions = source_dataset.clipped_actions
    env_param_src = source_dataset.source_param[:, 1+env_param.param_space_size:1+env_param.param_space_size+env_param.env_param_space_size]
    trajectories_length = source_dataset.source_param[:, 1+env_param.param_space_size+env_param.env_param_space_size]

    return [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions, trajectories_length]


def computeImportanceWeightsSourceTarget(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_action_t, trajectories_length] = getEpisodesInfoFromSource(source_dataset, env_param)
    variance_action = simulation_param.variance_action
    variance_env = env_param_src[:, -1]
    param_indices = np.concatenate(([0], np.cumsum(np.delete(source_dataset.episodes_per_config, -1))))
    state_t1_denoised_current = env_param.env.stepDenoisedCurrent(state_t, clipped_action_t) #change to source dataset . states denoised
    state_t1_denoised = source_dataset.next_states_unclipped_denoised

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
        policy_tgt[mask] = 0
        policy_src[mask] = 0
        model_tgt[mask] = 0
        model_src[mask] = 0

        policy_tgt = np.cumprod(policy_tgt, axis=1)
        policy_src = np.cumprod(policy_src, axis=1)

        model_tgt = np.cumprod(model_tgt, axis=1)
        model_src = np.cumprod(model_src, axis=1)


    weights = policy_tgt / policy_src * model_tgt / model_src

    return [weights, 0]


def computeMultipleImportanceWeightsSourceTarget(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions, trajectories_length] = getEpisodesInfoFromSource(source_dataset, env_param)
    variance_action = simulation_param.variance_action

    n = state_t.shape[0]
    state_t1_denoised_current = env_param.env.stepDenoisedCurrent(state_t, clipped_actions)
    variance_env = env_param_src[:, -1] # variance of the model transition

    policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2))**2)/(2*variance_action))
    model_tgt = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1 - state_t1_denoised_current)**2, axis=2) / (2*variance_env[:, np.newaxis]))

    policy_tgt[source_dataset.mask_weights] = 1
    model_tgt[source_dataset.mask_weights] = 1

    policy_tgt = np.prod(policy_tgt, axis=1)
    model_tgt = np.prod(model_tgt, axis=1)

    mis_denominator = np.squeeze(np.asarray(np.sum(np.multiply(source_dataset.episodes_per_config[np.newaxis, :]/n, source_dataset.source_distributions), axis=1)))

    weights = policy_tgt * model_tgt / mis_denominator

    return [weights, mis_denominator]


def computeMultipleImportanceWeightsSourceTargetPerDecision(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, batch_size):

    [param_policy_src, state_t, state_t1, unclipped_action_t, env_param_src, clipped_actions, trajectories_length] = getEpisodesInfoFromSource(source_dataset, env_param)
    variance_action = simulation_param.variance_action

    n = state_t.shape[0]

    state_t1_denoised_current = env_param.env.stepDenoisedCurrent(state_t, clipped_actions)
    variance_env = env_param_src[:, -1] # variance of the model transition

    policy_tgt = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply(policy_param[np.newaxis, np.newaxis, :], state_t), axis=2))**2)/(2*variance_action))
    model_tgt = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((state_t1 - state_t1_denoised_current)**2, axis=2) / (2*variance_env[:, np.newaxis]))

    policy_tgt[source_dataset.mask_weights] = 0
    model_tgt[source_dataset.mask_weights] = 0

    policy_tgt = np.cumprod(policy_tgt, axis=1)
    model_tgt = np.cumprod(model_tgt, axis=1)

    mis_denominator = np.sum(np.multiply(source_dataset.episodes_per_config[np.newaxis, np.newaxis, :]/n, source_dataset.source_distributions), axis=2)

    weights = policy_tgt * model_tgt / mis_denominator

    return [weights, mis_denominator]


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
        src_distributions_policy[mask] = 1

        src_distributions_policy = np.prod(src_distributions_policy, axis=1)
        src_distributions_model = np.prod(src_distributions_model, axis=1)

        source_dataset.mask_weights = mask

    else:
        src_distributions_policy = 1/m.sqrt(2*m.pi*variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, :], state_t), axis=2))**2)/(2*variance_action))
        src_distributions_model = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum(((state_t1 - state_t1_denoised)**2), axis=2) / (2*variance_env[:, np.newaxis, np.newaxis]))

        mask = trajectories_length[:, np.newaxis] < np.repeat(np.arange(0, state_t.shape[1])[np.newaxis, :], repeats=state_t.shape[0], axis=0)
        src_distributions_policy[mask] = 0
        src_distributions_model[mask] = 0

        src_distributions_policy = np.cumprod(src_distributions_policy, axis=1)
        src_distributions_model = np.cumprod(src_distributions_model, axis=1)

        source_dataset.mask_weights = mask

    return src_distributions_model * src_distributions_policy


def computeEss(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration):

    trajectories_length = getEpisodesInfoFromSource(source_dataset, env_param)[-1]
    n = trajectories_length.shape[0]

    weights = algorithm_configuration.computeWeights(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, 0)[0]

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

    weights = algorithm_configuration.computeWeights(policy_param, env_param, source_dataset, simulation_param, algorithm_configuration, 0)[0]

    print("Mean weights: " + str(np.mean(weights)))

    if algorithm_configuration.pd == 0:
        variance_weights = 1/n * np.sum((weights-1)**2)
        ess = n / (1 + variance_weights)
        min_index = 0
    else:
        ess = np.zeros(weights.shape[1])
        for t in range(weights.shape[1]):
            indices = trajectories_length >= t
            weights_timestep = weights[indices, t]
            variance_weights = 1/n * np.sum((weights_timestep-1)**2)
            ess[t] = n / (1 + variance_weights)

        min_index = np.argmin(ess)
        ess = ess[min_index]

    return [ess, min_index]


def computeNdef(min_index, param, env_param, source_dataset, simulation_param, algorithm_configuration):

    trajectories_length = getEpisodesInfoFromSource(source_dataset, env_param)[-1]
    weights = algorithm_configuration.computeWeights(param, env_param, source_dataset, simulation_param, algorithm_configuration, 1)[0]
    n = source_dataset.source_task.shape[0]

    if algorithm_configuration.pd == 1:
        indices = trajectories_length >= min_index
        weights = weights[indices, min_index]

    w_1 = np.linalg.norm(weights, 1)
    w_2 = np.linalg.norm(weights, 2)
    num_episodes_target1 = int(max(1, np.ceil((simulation_param.ess_min * w_1 / n) - (w_1 * n / (w_2 ** 2)))))

    return num_episodes_target1


def computeNdefSecond(min_index, param, env_param, source_dataset, simulation_param, algorithm_configuration):

    trajectories_length = getEpisodesInfoFromSource(source_dataset, env_param)[-1]
    weights = algorithm_configuration.computeWeights(param, env_param, source_dataset, simulation_param, algorithm_configuration, 1)[0]
    n = source_dataset.source_task.shape[0]

    if algorithm_configuration.pd == 1:
        indices = trajectories_length >= min_index
        weights = weights[indices, min_index]

    variance_weights = 1/n * np.sum((weights-1)**2)
    c = (np.mean(weights**3) + 3*(1-np.mean(weights)))/(1 + variance_weights)**2
    num_episodes_target2 = np.ceil((simulation_param.ess_min - n / (1 + np.var(weights)))/(min(1, c)))
    num_episodes_target2 = int(np.clip(num_episodes_target2, 1, simulation_param.ess_min))

    return num_episodes_target2


def essPerTarget(env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, source_dataset, simulation_param, algorithm_configuration, env_params):

    policy_param = np.linspace(policy_param_min, policy_param_max, linspace_policy)
    env_parameters = np.linspace(env_param_min, env_param_max, linspace_env)
    ess1 = np.zeros((env_parameters.shape[0], policy_param.shape[0]))
    ess2 = np.zeros((env_parameters.shape[0], policy_param.shape[0]))
    n_def1 = np.zeros((env_parameters.shape[0], policy_param.shape[0]))
    n_def2 = np.zeros((env_parameters.shape[0], policy_param.shape[0]))

    for i_policy_param in range(policy_param.shape[0]):
        print(i_policy_param)
        for i_env_param in range(env_parameters.shape[0]):

            env_params.env.setParams(np.concatenate(([env_parameters[i_env_param]], np.ravel(env_params.env.B), [env_params.env.sigma_noise**2])))
            print(str(i_env_param) + " " + str(i_policy_param))
            [ess1_ij, min_index1] = computeEss(np.asarray(policy_param[i_policy_param] * np.ones(env_param.param_space_size)), env_params, source_dataset, simulation_param, algorithm_configuration)
            n_def1_ij = computeNdef(min_index1, np.asarray(policy_param[i_policy_param] * np.ones(env_param.param_space_size)), env_params, source_dataset, simulation_param, algorithm_configuration)
            ess1[i_env_param, i_policy_param] = ess1_ij
            n_def1[i_env_param, i_policy_param] = n_def1_ij

            [ess2_ij, min_index2] = computeEssSecond(np.asarray(policy_param[i_policy_param] * np.ones(env_param.param_space_size)), env_params, source_dataset, simulation_param, algorithm_configuration)
            n_def2_ij = computeNdefSecond(min_index2, np.asarray(policy_param[i_policy_param] * np.ones(env_param.param_space_size)), env_params, source_dataset, simulation_param, algorithm_configuration)
            ess2[i_env_param, i_policy_param] = ess2_ij
            n_def2[i_env_param, i_policy_param] = n_def2_ij

    return [ess1, n_def1, ess2, n_def2]


env = gym.make("LQG1D-v0")
param_space_size = 1
state_space_size = 1
env_param_space_size = 3
episode_length = 20

env_param = EnvParam(env, param_space_size, state_space_size, env_param_space_size, episode_length)

mean_initial_param = 0
variance_initial_param = 0
variance_action = 0.1

batch_size = 20
discount_factor = 0.99
ess_min = 70

env_param_min = 0.6
env_param_max = 1.5
policy_param_min = -1
policy_param_max = -0.1
linspace_env = 5
linspace_policy = 5

[source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised] = stc.sourceTaskCreationAllCombinations(env, episode_length, batch_size, discount_factor, variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env, linspace_policy, param_space_size, state_space_size, env_param_space_size)
source_dataset = SourceDataset(source_task, source_param, episodes_per_configuration, next_states_unclipped, actions_clipped, next_states_unclipped_denoised)

simulation_param = SimulationParam(mean_initial_param, variance_initial_param, variance_action, batch_size, discount_factor, ess_min)

#Estimator details

pd = 0
computeWeights = computeMultipleImportanceWeightsSourceTarget

algorithm_configuration = AlgorithmConfiguration(pd, computeWeights)

source_dataset.source_distributions = computeMultipleImportanceWeightsSourceDistributions(source_dataset, variance_action, algorithm_configuration, env_param)

print("Computing ESS")
[ess1, n_def1, ess2, n_def2] = essPerTarget(env_param_min, env_param_max, policy_param_min, policy_param_max, linspace_env*2, linspace_policy*2, source_dataset, simulation_param, algorithm_configuration, env_param)


with open('ess_version1.pkl', 'wb') as output:
    pickle.dump(ess1, output, pickle.HIGHEST_PROTOCOL)

with open('ess_version2.pkl', 'wb') as output:
    pickle.dump(ess2, output, pickle.HIGHEST_PROTOCOL)

with open('n_def_version1.pkl', 'wb') as output:
    pickle.dump(n_def1, output, pickle.HIGHEST_PROTOCOL)

with open('n_def_version2.pkl', 'wb') as output:
    pickle.dump(n_def2, output, pickle.HIGHEST_PROTOCOL)

print(ess2)

average = np.mean(ess1)
print(np.max(ess1))
ax = sns.heatmap(ess1, linewidth=0.5, vmax=np.max(ess1), vmin=np.min(ess1), center=average)
plt.show()

average = np.mean(ess2)
print(np.max(ess2))
ax = sns.heatmap(ess2, linewidth=0.5, vmax=np.max(ess2), vmin=np.min(ess2), center=average)
plt.show()

average = np.mean(n_def1)
print(np.max(n_def1))
ax = sns.heatmap(n_def1, linewidth=0.5, vmax=np.max(n_def1), vmin=np.min(n_def1), center=average)
plt.show()

average = np.mean(n_def2)
print(np.max(n_def2))
ax = sns.heatmap(n_def2, linewidth=0.5, vmax=np.max(n_def2), vmin=np.min(n_def2), center=average)
plt.show()
