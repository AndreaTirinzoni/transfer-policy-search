import numpy as np
import simulationClasses as sc

class SourceEstimator:

    def __init__(self, source_dataset, model_estimator_list):
        self.n_models = int(source_dataset.episodes_per_config.shape[0]/source_dataset.policy_per_model)
        self.transition_models = model_estimator_list
        t = 0
        j = 0
        for i in range(self.n_models):
            episodes_per_model = np.sum(source_dataset.episodes_per_config[j:j+source_dataset.policy_per_model])
            source_task = source_dataset.source_task[t:t+episodes_per_model, :, :]
            episodes_per_config = source_dataset.episodes_per_config[j:j+source_dataset.policy_per_model]
            clipped_actions = source_dataset.clipped_actions[t:t+episodes_per_model, :]
            source_param = source_dataset.source_param[t:t+episodes_per_model, :]
            next_states_unclipped = source_dataset.next_states_unclipped[t:t+episodes_per_model, :, :]
            next_states_unclipped_denoised = source_dataset.next_states_unclipped_denoised[t:t+episodes_per_model, :, :]
            n_config_cv = source_dataset.n_config_cv
            source_dataset_current_model = sc.SourceDataset(source_task, source_param, episodes_per_config, next_states_unclipped, clipped_actions, next_states_unclipped_denoised, n_config_cv)
            self.transition_models[i].update_model(source_dataset_current_model, source_task=True)
            t += episodes_per_model
            j += source_dataset.policy_per_model

    def stepDenoised(self, state, action, policy_per_model):
        state_t1_denoised = np.zeros(state.shape)
        state_t1_denoised = np.repeat(state_t1_denoised[:, :, :, np.newaxis], self.n_models*policy_per_model, axis=3)
        t = 0
        for i in range(self.n_models):
            state_t1_denoised[:, :, :, t:t+policy_per_model] = self.transition_models[i].transition(state, action)[:, :, :, np.newaxis]
            t += policy_per_model
        return state_t1_denoised

    def stepDenoisedSingle(self, state, action, policy_per_model):
        state_t1_denoised = np.zeros(state.shape)
        t = 0
        for i in range(self.n_models):
            state_t1_denoised[t:t+policy_per_model, :, :] = self.transition_models[i].transition(state[t:t+policy_per_model, :, :], action[t:t+policy_per_model, :, :])
            t += policy_per_model
        return state_t1_denoised

    def density(self, state, action, state_t1, policy_per_model):
        density_funct = np.zeros((state.shape[0], state.shape[1]))
        density_funct = np.repeat(density_funct[:, :, np.newaxis], self.n_models*policy_per_model, axis=2)
        t = 0
        for i in range(self.n_models):
            density_funct[:, :, t:t+policy_per_model] = self.transition_models[i].density(state, action, state_t1)[:, :, np.newaxis]
            t += policy_per_model
        return density_funct

    def singleDensity(self, state, action, state_t1, policy_per_model):
        density_funct = np.zeros((state.shape[0], state.shape[1]))
        t = 0
        for i in range(self.n_models):
            density_funct[t:t+policy_per_model, :] = self.transition_models[i].density(state[t:t+policy_per_model, :, :], action[t:t+policy_per_model, :, :], state_t1[t:t+policy_per_model, :, :])
            t += policy_per_model
        return density_funct
