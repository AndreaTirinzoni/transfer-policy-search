import numpy as np
import sourceTaskCreation as stc
import math as m
import simulation as simu

class Models:

    def __init__(self, proposal_env):

        self.proposal_env = proposal_env
        self.probabilities_env = np.ones(proposal_env.shape[0]) / proposal_env.shape[0]
        env_parameter_proposal = []
        for i in range(proposal_env.shape[0]):
            env_parameter_proposal.append(proposal_env[i].getEnvParam.T)

        self.env_parameters_proposal = env_parameter_proposal


    def computeModelsProbability(self, dataset_model_estimation, env_params, param_policy, simulation_param):

        env_probabilities = self.probabilities_env

        state_t = dataset_model_estimation.source_task[:, :, 0:env_params.state_space_size] # state t
        state_t1 = dataset_model_estimation.next_states_unclipped # state t+1
        unclipped_action_t = dataset_model_estimation.source_task[:, :, env_params.state_space_size]
        clipped_actions = dataset_model_estimation.clipped_actions
        trajectories_length = dataset_model_estimation.source_param[:, 1+env_params.param_space_size+env_params.env_param_space_size]


        #for every proposal compute the new probabilities
        for i in range(env_probabilities.shape[0]):

            env = self.proposal_env[i]
            env_param_src = env.getEnvParam()
            variance_env = env_param_src[-1]
            state_t1_denoised_current = env.stepDenoisedCurrent(state_t, clipped_actions)
            policy_transition = 1/m.sqrt(2*m.pi*simulation_param.variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(param_policy[np.newaxis, np.newaxis, :], state_t), axis=2)))**2)/(2*simulation_param.variance_action))
            model_transition = 1/np.sqrt((2*m.pi*variance_env)**env_params.state_space_size) * np.exp(-(np.sum((state_t1 - state_t1_denoised_current)**2, axis=2)) / (2*variance_env))

            for t in range(state_t.shape[1]):
                indices = trajectories_length < t
                policy_transition[indices, t] = 1
                model_transition[indices, t] = 1

            policy_transition = np.prod(policy_transition, axis=1)
            model_transition = np.prod(model_transition, axis=1)

            env_probabilities[i] = env_probabilities[i] * np.prod(policy_transition * model_transition)

        self.probabilities_env = env_probabilities


    def getEpisodesInfoFromSource(self, source_dataset, env_param):

        param_policy_src = source_dataset.source_param[:, 1:1+env_param.param_space_size] # policy parameter of source
        state_t = source_dataset.source_task[:, :, 0:env_param.state_space_size] # state t
        state_t1 = source_dataset.next_states_unclipped # state t+1
        unclipped_action_t = source_dataset.source_task[:, :, env_param.state_space_size]
        clipped_actions = source_dataset.clipped_actions
        env_param_src = source_dataset.source_param[:, 1+env_param.param_space_size:1+env_param.param_space_size+env_param.env_param_space_size]
        trajectories_length = source_dataset.source_param[:, 1+env_param.param_space_size+env_param.env_param_space_size]

        return [state_t, state_t1, unclipped_action_t, clipped_actions, trajectories_length, param_policy_src, env_param_src]


    def computeExpectedValueCurrentProposal(self, env, env_param, param_policy, simulation_param, source_parameters, episodes_per_config, initial_size):

        batch_size = 50
        dataset_current_env = stc.sourceTaskCreationSpec(env, env_param.episode_length, batch_size, simulation_param.discount_factor, simulation_param.variance_action, param_policy, env.getEnvParam(), env_param.param_space_size, env_param.state_space_size, env_param.env_param_space_size)
        dataset_current_env = simu.SourceDataset(dataset_current_env)

        [state_t, state_t1, unclipped_action_t, clipped_actions, trajectories_length] = self.getEpisodesInfoFromSource(dataset_current_env, env_param)[0:5]

        source_parameters[initial_size:, 1+env_param.param_space_size:1+env_param.param_space_size+env_param.env_param_space_size] = env.getEnvParam().T

        param_indices = np.concatenate(([0], np.cumsum(np.delete(episodes_per_config, -1))))
        env_param_src = source_parameters[:, 1+env_param.param_space_size:1+env_param.param_space_size+env_param.env_param_space_size]
        param_policy_src = source_parameters[:, 1:1+env_param.param_space_size]

        combination_src_parameters = (param_policy_src[param_indices, :])
        combination_src_parameters_env = (env_param_src[param_indices, :])#policy parameter of source not repeated

        state_t = np.repeat(state_t[:, :, :, np.newaxis], combination_src_parameters.shape[0], axis=3) # state t
        state_t1 = np.repeat(state_t1[:, :, :, np.newaxis], combination_src_parameters.shape[0], axis=3) # state t+1
        unclipped_action_t = np.repeat(unclipped_action_t[:, :, np.newaxis], combination_src_parameters.shape[0], axis=2) # action t
        clipped_actions = np.repeat(clipped_actions[:, :, np.newaxis], combination_src_parameters_env.shape[0], axis=2) # action t
        variance_env = env_param_src[:, -1] # variance of the model transition
        state_t1_denoised = env.stepDenoised(combination_src_parameters_env, state_t, clipped_actions)
        state_t1_denoised_current = env.stepDenoisedCurrent(state_t, clipped_actions)

        mis_distributions_policy = 1/m.sqrt(2*m.pi*simulation_param.variance_action) * np.exp(-((unclipped_action_t - np.sum(np.multiply((combination_src_parameters.T)[np.newaxis, np.newaxis, :, :], state_t), axis=2))**2)/(2*simulation_param.variance_action))
        mis_distributions_model = 1/np.sqrt((2*m.pi*variance_env[:, np.newaxis, np.newaxis])**env_param.state_space_size) * np.exp(-np.sum((np.power((state_t1 - state_t1_denoised), 2)), axis=2) / (2*variance_env[:, np.newaxis, np.newaxis]))

        mask = np.ones((mis_distributions_policy.shape[0], mis_distributions_policy.shape[1]))
        for t in range(mis_distributions_policy.shape[1]):
            indices = trajectories_length < t
            if indices[indices == True].shape[0] > 0:
                mask[indices, t] = 1/(mis_distributions_model[indices, t, 0] * mis_distributions_policy[indices, t, 0])

        src_distributions_policy = np.prod(mis_distributions_policy * mask[:, :, np.newaxis], axis=1)
        src_distributions_model = np.prod(mis_distributions_model, axis=1)

        q_j = src_distributions_model * src_distributions_policy

        policy_transition = 1/m.sqrt(2*m.pi*simulation_param.variance_action) * np.exp(-((unclipped_action_t - (np.sum(np.multiply(param_policy[np.newaxis, np.newaxis, :], state_t), axis=2)))**2)/(2*simulation_param.variance_action))
        model_transition = 1/np.sqrt((2*m.pi*variance_env)**env_param.state_space_size) * np.exp(-(np.sum((state_t1 - state_t1_denoised_current)**2, axis=2)) / (2*variance_env))

        for t in range(state_t.shape[1]):
            indices = trajectories_length < t
            policy_transition[indices, t] = 1
            model_transition[indices, t] = 1

        policy_transition = np.prod(policy_transition, axis=1)
        model_transition = np.prod(model_transition, axis=1)

        n = dataset_current_env.initial_size

        mis_denominator = np.squeeze(np.asarray(np.sum(np.multiply(dataset_current_env.episodes_per_config[np.newaxis, :]/n, q_j), axis=1)))

        return np.mean(policy_transition * model_transition / mis_denominator)


    def computeExpectedValueMixture(self, env, env_param, param_policy, simulation_param, source_parameters, episodes_per_config, initial_size):

        source_parameters[:, 1+env_param.param_space_size:1+env_param.param_space_size+env_param.env_param_space_size] = env.getEnvParam().T

        param_indices = np.concatenate(([0], np.cumsum(np.delete(episodes_per_config[initial_size:], -1))))
        param_policy_src = source_parameters[:, 1:1+env_param.param_space_size]

        combination_src_parameters = (param_policy_src[param_indices, :])

        batch_size = 50
        dataset_mixture = stc.sourceTaskCreationMixture(env, env_param.episode_length, batch_size, simulation_param.discount_factor, simulation_param.variance_action, combination_src_parameters, episodes_per_config, initial_size, env_param.param_space_size, env_param.state_space_size, env_param.env_param_space_size)
        dataset_mixture = simu.SourceDataset(dataset_mixture)

        [state_t, state_t1, unclipped_action_t, clipped_actions, trajectories_length] = self.getEpisodesInfoFromSource(dataset_mixture, env_param)[0:5]

        combination_src_parameters_env = self.env_parameters_proposal # policy parameter of source not repeated

        state_t = np.repeat(state_t[:, :, :, np.newaxis], combination_src_parameters_env.shape[0], axis=3) # state t
        clipped_actions = np.repeat(clipped_actions[:, :, np.newaxis], combination_src_parameters_env.shape[0], axis=2) # action t
        state_t1_denoised = env.stepDenoised(combination_src_parameters_env, state_t, clipped_actions)
        state_t1_denoised_current = env.stepDenoisedCurrent(state_t, clipped_actions)

        mask = np.ones((state_t.shape[0], state_t.shape[1]))
        for t in range(state_t.shape[1]):
            indices = trajectories_length < t
            if indices[indices == True].shape[0] > 0:
                mask[indices, t] = 0

        m = np.sum(self.probabilities_env[:, :, :, np.newaxis] * state_t1_denoised, axis=4)
        norm = np.linalg.norm((m-state_t1_denoised_current)**2, 2, axis=3)

        return np.sum(np.mean(norm, axis=0), axis=0)


    def computeLossFunction(self, env_params, param_policy, simulation_param, source_parameters, episodes_per_config, initial_size):

        loss_function = np.zeros(self.proposal_env.shape[0])
        n = source_parameters.shape[0]

        for i in range(self.probabilities_env.shape[0]):
            env = self.proposal_env[i]
            expected_value_current_proposal = self.computeExpectedValueCurrentProposal(env, env_params, param_policy, simulation_param, source_parameters, episodes_per_config, initial_size)
            expected_value_mixture = self.computeExpectedValueMixture(env, env_params, param_policy, simulation_param, source_parameters[initial_size:], episodes_per_config, initial_size)
            alpha_0 = episodes_per_config[-1] / n
            alpha_tr = np.sum(episodes_per_config[initial_size:]) / np.sum(episodes_per_config)
            sigma = simulation_param.variance_action
            loss_function[i] = 1/n * expected_value_current_proposal + 4 * alpha_tr / (alpha_0**2 * sigma) * expected_value_mixture

        return loss_function


    def chooseTransitionModel(self, env_params, param_policy, simulation_param, source_parameters, episodes_per_config, initial_size):
        batch_size = 50
        dataset_model_estimation = stc.sourceTaskCreationSpec(env_params.env_tgt, env_params.episode_length, batch_size, simulation_param.discount_factor, simulation_param.variance_action, param_policy, env_params.env_tgt.getEnvParam(), env_params.param_space_size, env_params.state_space_size, env_params.env_param_space_size)
        dataset_model_estimation = simu.SourceDataset(dataset_model_estimation)
        self.computeModelsProbability(dataset_model_estimation, env_params, param_policy, simulation_param)
        lossFunction = self.computeLossFunction(env_params, param_policy, simulation_param, source_parameters, episodes_per_config, initial_size)
        min_index = np.argmin(lossFunction)
        env = self.proposal_env[min_index]

        return env
