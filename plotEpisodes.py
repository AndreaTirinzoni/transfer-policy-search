import pickle
from utils import plot
import numpy as np
import matplotlib.pyplot as plt


def batchToEpisodes(statistic_batch, episodesPerBatch, max_episodes):

    statistic_episode = np.ones(max_episodes)

    initial_index = 0

    for i in range(episodesPerBatch.shape[0]):
        episode_current_batch = episodesPerBatch[i]
        statistic_episode[initial_index:initial_index+episode_current_batch] = statistic_batch[i]
        initial_index += episode_current_batch

    statistic_episode[initial_index:] = statistic_batch[i]

    return statistic_episode

with open('results.pkl', 'rb') as input:
    results = pickle.load(input)

#estimators = ["PD-IS", "IS", "PD-MIS", "PD-MIS-CV-BASELINE", "MIS", "MIS-CV-BASELINE", "GPOMDP"] #minigolf
#estimators = ["GPOMDP", "PD-IS", "PD-MIS-CV-BASELINE", "PD-MIS-CV-BASELINE-SR"] #cartpole
# estimators = ["IS", "PD-IS", "MIS", "MIS-CV-BASELINE", "PD-MIS", "PD-MIS-CV-BASELINE", "GPOMDP"] #lqg1d #lqg1d
# #
# runs = 20
# linspace_episodes = 10
# param_policy_space = 1
#
# n_def = results[0]["GPOMDP"][0].n_def
# max_episodes = int(np.sum(n_def))
# # stats_together = 0
#
# disc_rewards = {}
# tot_rewards = {}
# policy = {}
# gradient = {}
# ess = {}
# n_def = {}
#
# for estimator in estimators:
#     disc_rewards[estimator] = []
#     tot_rewards[estimator] = []
#     policy[estimator] = []
#     gradient[estimator] = []
#     ess[estimator] = []
#     n_def[estimator] = []
#
# x = np.asarray(range(int(max_episodes)))
#
# from scipy.stats import t
# alpha = t.interval(0.95, runs-1, loc=0, scale=1)[1] if runs > 1 else 1
#
#
# for estimator in estimators:
#     for i in range(runs):
#         disc_rewards_current_run = results[i][estimator][0].disc_rewards
#         episodes_current_run = results[i][estimator][0].n_def
#         disc_rewards_current_run_episodes = batchToEpisodes(disc_rewards_current_run, episodes_current_run.astype(int), max_episodes)
#         disc_rewards[estimator].append(disc_rewards_current_run_episodes)
#         tot_rewards_current_run = results[i][estimator][0].total_rewards
#         episodes_current_run = results[i][estimator][0].n_def
#         tot_rewards_current_run_episodes = batchToEpisodes(tot_rewards_current_run, episodes_current_run.astype(int), max_episodes)
#         tot_rewards[estimator].append(tot_rewards_current_run_episodes)
#         n_def_current_run = results[i][estimator][0].n_def
#         episodes_current_run = results[i][estimator][0].n_def
#         n_def[estimator].append(episodes_current_run)
#         policy_current_run = results[i][estimator][0].policy_parameter
#         policy_current_run_episodes = []
#         for t in range(param_policy_space):
#             policy_current_run_episodes.append(np.asarray([batchToEpisodes(policy_current_run[:, t],
#                                                                            episodes_current_run.astype(int),
#                                                                            max_episodes)]))
#         policy[estimator].append(policy_current_run_episodes)
#
#
# #Plotting the rewards
#
# means = [np.mean(tot_rewards[estimator], axis=0)[0::linspace_episodes] for estimator in estimators]
# stds = [alpha * np.std(tot_rewards[estimator], axis=0)[0::linspace_episodes] / np.sqrt(runs) for estimator in estimators]
#
# plot.plot_curves([x[0::linspace_episodes] for _ in estimators], means, stds, x_label="Episodes", y_label="Return", names=estimators)
#
# #Plotting the policies
# means = [np.mean(policy[estimator], axis=0)[:, :, 0::linspace_episodes] for estimator in estimators]
# stds = [alpha * np.std(policy[estimator], axis=0)[:, :, 0::linspace_episodes] / np.sqrt(runs) for estimator in estimators]
#
# for i in range(param_policy_space):
#     plot.plot_curves([x[0::linspace_episodes] for _ in estimators], [np.asarray(np.squeeze(means[estimator][i, :])) for estimator in range(len(estimators))], [np.asarray(np.squeeze(stds[estimator][i, :])) for estimator in range(len(estimators))], x_label="Episodes", y_label="Policy", names=estimators)
#
# #Plotting n_def
# means = [np.mean(n_def[estimator], axis=0) for estimator in estimators]
# stds = [alpha * np.std(n_def[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]
#
# num_batch = results[0]["GPOMDP"][0].n_def.shape[0]
# x = range(num_batch)
#
# plot.plot_curves([x for _ in estimators], means, stds, x_label="Episodes", y_label="n_def", names=estimators)
#
#
# # Plot disc rewards
# fig, ax = plt.subplots()
#
# plt.style.use('ggplot')
#
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
# plt.rcParams['font.size'] = 20
# plt.rcParams['axes.labelsize'] = 20
# # plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titlesize'] = 20
# plt.rcParams['xtick.labelsize'] = 16
# plt.rcParams['ytick.labelsize'] = 16
# plt.rcParams['legend.fontsize'] = 22
# plt.rcParams['figure.titlesize'] = 20
#
#
# X = np.array(range(max_episodes))
# plt.xlim([X.min(), X.max()])
#
# x = [range(max_episodes) for _ in range(runs)]
#
# for i in range(len(x)):
#     #for t in range(param_policy_space):
#     ax.plot(x[i], policy["PD-MIS-CV-BASELINE"][i][0][0, :])
#
# plt.show()
#
#
# x = [range(max_episodes) for _ in range(runs)]
#
# for i in range(len(x)):
#
#     ax.plot(x[i], tot_rewards["PD-MIS-CV-BASELINE"][i][0])

#plt.show()
runs = 3

t = 0

n_def1 = np.array([1, 2, 3])
n_def2 = np.array([1, 2, 3])
n_def3 = np.array([1, 2, 3])
rew1 = np.array([1, 2, 3])
rew2 = np.array([2, 3, 4])
rew3 = np.array([3, 4, 5])

n_def2_tot = np.sum(n_def1)

n_def12 = np.ones(n_def1[0])*rew1[0]
n_def22 = np.ones(n_def1[0])*rew2[0]
n_def32 = np.ones(n_def1[0])*rew3[0]

for i in range(1, n_def1.shape[0]):
    n_def12 = np.concatenate((n_def12, np.ones(n_def1[i])*rew1[i]), axis=0)
    n_def22 = np.concatenate((n_def22, np.ones(n_def2[i])*rew2[i]), axis=0)
    n_def32 = np.concatenate((n_def32, np.ones(n_def3[i])*rew3[i]), axis=0)

max_episodes = np.max(np.asarray([np.sum(n_def1), np.sum(n_def2), np.sum(n_def3)]))
rew_ep1 = batchToEpisodes(rew1, n_def1, max_episodes)
rew_ep2 = batchToEpisodes(rew2, n_def2, max_episodes)
rew_ep3 = batchToEpisodes(rew3, n_def3, max_episodes)

min = np.min([n_def12.shape[0], n_def22.shape[0], n_def32.shape[0]])

x = range(min)

mean1 = np.mean(np.asarray([n_def12[0:min], n_def22[0:min], n_def32[0:min]]), axis=0)
std1 = np.std(np.asarray([n_def12[0:min], n_def32[0:min], n_def32[0:min]]), axis=0)

mean2 = np.mean(np.asarray([rew_ep1, rew_ep2, rew_ep3]), axis=0)[0:min]
std2 = np.std(np.asarray([rew_ep1, rew_ep2, rew_ep3]), axis=0)[0:min]


# plt.plot(x, n_def12[0:min])
# plt.plot(x, n_def22[0:min])
# plt.plot(x, n_def32[0:min])
# plt.plot(x, rew_ep1[0:min])
# plt.plot(x, rew_ep2[0:min])
# plt.plot(x, rew_ep3[0:min])

plt.plot(np.cumsum(n_def1), rew1)
plt.plot(np.cumsum(n_def2), rew2)
plt.plot(np.cumsum(n_def3), rew3)

plt.show()

plt.plot(x, mean2[0:min])
plt.show()
