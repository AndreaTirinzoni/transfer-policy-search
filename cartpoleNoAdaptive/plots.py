import pickle
from utils import plot
import numpy as np


def batchToEpisodes(statistic_batch, episodesPerBatch, linspace_episodes, max_episodes):

    statistic_episode = np.ones(max_episodes)

    initial_index = 0

    for i in range(episodesPerBatch.shape[0]):
        episode_current_batch = episodesPerBatch[i]
        statistic_episode[initial_index:initial_index+episode_current_batch] = statistic_batch[i]
        initial_index += episode_current_batch

    statistic_episode[initial_index:] = statistic_batch[i]

    return statistic_episode[0::linspace_episodes]

with open('results.pkl', 'rb') as input:
    results = pickle.load(input)

estimators = ["PD-IS", "PD-MIS", "PD-MIS-CV-BASELINE-APPROXIMATED", "GPOMDP"]

disc_rewards = {}
tot_rewards = {}
policy = {}
gradient = {}
ess = {}
n_def = {}

for estimator in estimators:
    disc_rewards[estimator] = []
    tot_rewards[estimator] = []
    policy[estimator] = []
    gradient[estimator] = []
    ess[estimator] = []
    n_def[estimator] = []

runs = 20
linspace_episodes = 1
param_policy_space = 1
n_def = results[0]["GPOMDP"][0].n_def
max_episodes = int(np.sum(n_def))
stats_together = 0

x = range(int(max_episodes/linspace_episodes))

from scipy.stats import t
alpha = t.interval(0.95, runs-1, loc=0, scale=1)[1] if runs > 1 else 1

# if stats_together == 0:
#     for estimator in estimators:
#         for i in range(runs):
#             disc_rewards_current_run = disc_rewards[estimator][i]
#             episodes_current_run = n_def[estimator][i]
#             disc_rewards_current_run_episodes = batchToEpisodes(disc_rewards_current_run, episodes_current_run.astype(int), linspace_episodes, max_episodes)
#             disc_rewards[estimator].append(disc_rewards_current_run_episodes)
#             policy_current_run = policy[estimator][i]
#             episodes_current_run = n_def[estimator][i]
#             policy_current_run_episodes = [batchToEpisodes(policy_current_run[:, i], episodes_current_run.astype(int), linspace_episodes, max_episodes) for i in range(param_policy_space)]
#             policy[estimator].append(policy_current_run_episodes)
#
#     else:

for estimator in estimators:
    for i in range(runs):
        disc_rewards_current_run = results[i][estimator][0].disc_rewards
        episodes_current_run = results[i][estimator][0].n_def
        disc_rewards_current_run_episodes = batchToEpisodes(disc_rewards_current_run, episodes_current_run.astype(int), linspace_episodes, max_episodes)
        disc_rewards[estimator].append(disc_rewards_current_run_episodes)
        tot_rewards_current_run = results[i][estimator][0].total_rewards
        episodes_current_run = results[i][estimator][0].n_def
        tot_rewards_current_run_episodes = batchToEpisodes(tot_rewards_current_run, episodes_current_run.astype(int), linspace_episodes, max_episodes)
        tot_rewards[estimator].append(tot_rewards_current_run_episodes)
        policy_current_run = results[i][estimator][0].policy_parameter
        policy_current_run_episodes = [batchToEpisodes(policy_current_run[:, t], episodes_current_run.astype(int), linspace_episodes, max_episodes) for t in range(param_policy_space)]
        policy[estimator].append(policy_current_run_episodes)

means = [np.mean(tot_rewards[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(tot_rewards[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Return", names=estimators)

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

plot.plot_curves([x for _ in estimators], [means[estimator][0, :] for estimator in range(len(estimators))], [stds[estimator][0, :] for estimator in range(len(estimators))], x_label="Iteration", y_label="Policy", names=estimators)

means = [np.mean(n_def[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(n_def[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="N_DEF", names=estimators)
# means = [np.mean(rewards[estimator], axis=0) for estimator in estimators]
# stds = [alpha * np.std(rewards[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]
#
# plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Return", names=estimators)
