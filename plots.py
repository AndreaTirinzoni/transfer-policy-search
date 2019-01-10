import pickle
from utils import plot
import numpy as np


def batchToEpisodes(statistic_batch, episodesPerBatch, linspace_episodes, max_episodes):

    tot_episodes = np.sum(episodesPerBatch)
    statistic_episode = np.ones(max_episodes)

    initial_index = 0

    for i in range(episodesPerBatch.shape[0]):
        episode_current_batch = episodesPerBatch[i]
        statistic_episode[initial_index:initial_index+episode_current_batch] = statistic_batch[i]
        initial_index += episode_current_batch

    statistic_episode[initial_index:] = statistic_batch[i]

    return statistic_episode[0::linspace_episodes]

with open('rewards.pkl', 'rb') as input:
    rewards = pickle.load(input)

with open('policy.pkl', 'rb') as input:
    policy_batch = pickle.load(input)

with open('gradient.pkl', 'rb') as input:
    gradient = pickle.load(input)

with open('ess.pkl', 'rb') as input:
    ess = pickle.load(input)

with open('n_def.pkl', 'rb') as input:
    n_def = pickle.load(input)

estimators = ["MIS", "MIS-CV-BASELINE", "PD-MIS", "PD-MIS-CV-BASELINE", "GPOMDP"]
runs = 15
num_batch = 250
linspace_episodes = 1
param_policy_space = 1
max_episodes = int(np.sum(n_def["GPOMDP"][0]))

x = range(int(max_episodes/linspace_episodes))

from scipy.stats import t
alpha = t.interval(0.95, runs-1, loc=0, scale=1)[1] if runs > 1 else 1

disc_rewards = {}
policy = {}
for estimator in estimators:
    disc_rewards[estimator] = []
    policy[estimator] = []

for estimator in estimators:
    for i in range(runs):
        disc_rewards_current_run = rewards[estimator][i]
        episodes_current_run = n_def[estimator][i]
        disc_rewards_current_run_episodes = batchToEpisodes(disc_rewards_current_run, episodes_current_run.astype(int), linspace_episodes, max_episodes)
        disc_rewards[estimator].append(disc_rewards_current_run_episodes)
        policy_current_run = policy_batch[estimator][i]
        episodes_current_run = n_def[estimator][i]
        policy_current_run_episodes = [batchToEpisodes(policy_current_run[:, i], episodes_current_run.astype(int), linspace_episodes, max_episodes) for i in range(param_policy_space)]
        policy[estimator].append(policy_current_run_episodes)

# means = [np.mean(disc_rewards[estimator], axis=0) for estimator in estimators]
# stds = [alpha * np.std(disc_rewards[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]
#
# plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Return", names=estimators)

# means = [np.mean(ess[estimator], axis=0) for estimator in estimators]
# stds = [alpha * np.std(ess[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]
#
# plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Ess", names=estimators)

# means = [np.mean(gradient[estimator], axis=0) for estimator in estimators]
# stds = [alpha * np.std(gradient[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]
#
# plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Gradient", names=estimators)

means = [np.mean(policy[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(policy[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

plot.plot_curves([x for _ in estimators], [means[estimator][0, :] for estimator in range(5)], [stds[estimator][0, :] for estimator in range(5)], x_label="Iteration", y_label="Policy", names=estimators)

means = [np.mean(n_def[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(n_def[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="N_DEF", names=estimators)
# means = [np.mean(rewards[estimator], axis=0) for estimator in estimators]
# stds = [alpha * np.std(rewards[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]
#
# plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Return", names=estimators)
