import pickle
from utils import plot
import numpy as np

with open('results.pkl', 'rb') as input:
    results = pickle.load(input)

estimators = ["GPOMDP", "PD-IS", "PD-MIS-CV-BASELINE", "PD-MIS-CV-BASELINE-SR"] #cartpole
#estimators = ["PD-IS", "MIS", "MIS-CV-BASELINE", "PD-MIS", "PD-MIS-CV-BASELINE", "PD-MIS-CV-BASELINE_SR", "GPOMDP"] #lqg1d

runs = 20
linspace_episodes = 5
param_policy_space = 1
num_batch = results[0]["GPOMDP"][0].n_def.shape[0]

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


x = range(num_batch)

from scipy.stats import t
alpha = t.interval(0.95, runs-1, loc=0, scale=1)[1] if runs > 1 else 1


for estimator in estimators:
    for i in range(runs):
        disc_rewards_current_run = results[i][estimator][0].disc_rewards
        disc_rewards[estimator].append(disc_rewards_current_run)
        tot_rewards_current_run = results[i][estimator][0].total_rewards
        tot_rewards[estimator].append(tot_rewards_current_run)
        n_def_current_run = results[i][estimator][0].n_def
        n_def[estimator].append(n_def_current_run)
        policy_current_run = results[i][estimator][0].policy_parameter
        policy[estimator].append(policy_current_run)
        ess_current_run = results[i][estimator][0].ess
        ess[estimator].append(ess_current_run)

# Plot disc rewards
means = [np.mean(disc_rewards[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(disc_rewards[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Return", names=estimators)

# Plot tot rewards
means = [np.mean(tot_rewards[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(tot_rewards[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Return", names=estimators)

# Plot policies
means = [np.mean(policy[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(policy[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

for i in range(param_policy_space):
    plot.plot_curves([x for _ in estimators], [np.asarray(np.squeeze(means[estimator][:, i])) for estimator in range(len(estimators))], [np.asarray(np.squeeze(stds[estimator][:, i])) for estimator in range(len(estimators))], x_label="Iteration", y_label="Policy", names=estimators)

# Plot ess
means = [np.mean(n_def[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(n_def[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="n_def", names=estimators)

# Plot ess
means = [np.mean(ess[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(ess[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="ess", names=estimators)
