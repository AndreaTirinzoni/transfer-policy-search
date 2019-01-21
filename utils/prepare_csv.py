import numpy as np
import pickle
import sys
import os
import pandas as pd

sys.path.append("../../csv_files")


with open('./../results.pkl', 'rb') as input:
    results = pickle.load(input)

estimators = ["IS", "PD-IS", "MIS", "MIS-CV-BASELINE", "PD-MIS", "PD-MIS-CV-BASELINE", "GPOMDP"] #lqg1d

runs = 20
linspace_episodes = 1
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

#Store data per iteration

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
        gradient_current_run = results[i][estimator][0].gradient
        gradient[estimator].append(gradient_current_run)


# Base folder where to log
folder = "./../csv_files/lqg1d_adaptive"
os.makedirs(folder, exist_ok=True)
discounted_rewards_csv = []
total_rewards_csv = []
policy_csv = []
n_def_csv = []
ess_csv = []
gradient_csv = []
header = []



# Save disc rewards

means = [np.mean(disc_rewards[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(disc_rewards[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

discounted_rewards_csv.append(x)
header.append("iterations")

for i,estimator in zip(range(len(estimators)),estimators):
    discounted_rewards_csv.append(means[i])
    discounted_rewards_csv.append(stds[i])
    header.append(estimator + "-mean")
    header.append(estimator + "-std")

discounted_rewards_csv = np.asarray(discounted_rewards_csv).T
header = np.asarray(header)

df = pd.DataFrame(discounted_rewards_csv, columns=header)

df.to_csv(folder+"/discounted_rewards_lqg.csv")



# Save tot rewards
means = [np.mean(tot_rewards[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(tot_rewards[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

total_rewards_csv.append(x)

for i,estimator in zip(range(len(estimators)),estimators):
    total_rewards_csv.append(means[i])
    total_rewards_csv.append(stds[i])

total_rewards_csv = np.asarray(total_rewards_csv).T
header = np.asarray(header)

df = pd.DataFrame(total_rewards_csv, columns=header)

df.to_csv(folder+"/total_rewards_lqg.csv")


# Save policies
means = [np.mean(policy[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(policy[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

policy_csv.append(x)

for i in range(param_policy_space):
    for i,estimator in zip(range(len(estimators)),estimators):
        policy_csv.append(means[i])
        policy_csv.append(stds[i])

        policy_csv = np.asarray(policy_csv[:][:, i])
        df = pd.DataFrame(policy_csv, columns=header)
        df.to_csv(folder+"/policy" + str(i) + "_lqg.csv")


# Save n_def
means = [np.mean(n_def[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(n_def[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

n_def_csv.append(x)

for i,estimator in zip(range(len(estimators)),estimators):
    n_def_csv.append(means[i])
    n_def_csv.append(stds[i])

n_def_csv = np.asarray(n_def_csv).T

df = pd.DataFrame(n_def_csv, columns=header)

df.to_csv(folder+"/n_def_lqg.csv")


# Save ess
means = [np.mean(ess[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(ess[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

for i,estimator in zip(range(len(estimators)),estimators):
    ess_csv.append(means[i])
    ess_csv.append(stds[i])

ess_csv = np.asarray(ess_csv).T

df = pd.DataFrame(ess_csv, columns=header)

df.to_csv(folder+"/ess_csv_lqg.csv")


# Plot gradient
means = [np.mean(gradient[estimator], axis=0) for estimator in estimators]
stds = [alpha * np.std(gradient[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]

gradient_csv.append(x)

for i,estimator in zip(range(len(estimators)),estimators):
    gradient_csv.append(means[i])
    gradient_csv.append(stds[i])

gradient_csv = np.asarray(gradient_csv).T

df = pd.DataFrame(gradient_csv, columns=header)

df.to_csv(folder+"/gradient_csv_lqg.csv")
