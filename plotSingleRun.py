import pickle
from utils import plot
import numpy as np
import matplotlib.pyplot as plt

MARKERS = ["o", "D", "s", "^", "v", "p", "*"]
COLORS = ["#0e5ad3", "#bc2d14", "#22aa16", "#a011a3", "#d1ba0e", "#14ccc2", "#d67413", "#0e5ad3", "#bc2d14", "#22aa16", "#a011a3", "#d1ba0e", "#14ccc2", "#d67413", "#0e5ad3", "#bc2d14", "#22aa16", "#a011a3", "#d1ba0e", "#14ccc2", "#d67413"]
LINES = ["solid", "dashed", "dashdot", "dotted", "solid", "dashed", "dashdot", "dotted"]

with open('results.pkl', 'rb') as input:
    results = pickle.load(input)

#estimators = ["MIS-CV-BASELINE", "PD-MIS-CV-BASELINE", "GPOMDP"] #minigolf ["PD-MIS", "PD-MIS-CV-BASELINE-APPROXIMATED", "PD-MIS-CV-BASELINE", "GPOMDP"]
estimators = ["PD-MIS-CV-BASELINE"] #cartpole
#estimators = ["PD-IS", "MIS-CV-BASELINE", "PD-MIS", "PD-MIS-CV-BASELINE", "PD-MIS-CV-BASELINE-SR", "GPOMDP"] #lqg1d

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


x = [range(num_batch) for _ in range(runs)]

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
        gradient_current_run = results[i][estimator][0].gradient
        gradient[estimator].append(gradient_current_run)

# Plot disc rewards
fig, ax = plt.subplots()

plt.style.use('ggplot')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
# plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['figure.titlesize'] = 20

X = np.array(x)
plt.xlim([X.min(), X.max()])

# for i in range(len(x)):
#
#     ax.plot(x[i], tot_rewards["PD-MIS-CV-BASELINE"][i])

# Plot tot rewards

#plot.plot_curves([x for _ in range(runs)], tot_rewards["PD-MIS-CV-BASELINE"], np.zeros(num_batch), x_label="Iteration", y_label="Return", names=range(runs))

# Plot policies

for i in range(len(x)):

    ax.plot(x[i], policy["PD-MIS-CV-BASELINE"][i])

plt.show()
# for i in range(param_policy_space):
#     plot.plot_curves([x for _ in range(runs)], [policy["PD-MIS-CV-BASELINE"][t][:, i] for t in range(runs)], np.zeros(num_batch), x_label="Iteration", y_label="Policy", names=range(runs))

# Plot n_def

# plot.plot_curves([x for _ in range(runs)], n_def["PD-MIS-CV-BASELINE"], np.zeros(num_batch), x_label="Iteration", y_label="N_def", names=range(runs))
#
# # Plot ess
#
# plot.plot_curves([x for _ in range(runs)], ess["PD-MIS-CV-BASELINE"], np.zeros(num_batch), x_label="Iteration", y_label="Ess", names=range(runs))
#
# # Plot gradient
#
# for i in range(param_policy_space):
#     plot.plot_curves([x for _ in range(runs)], [gradient["PD-MIS-CV-BASELINE"][t][:, i] for t in range(runs)], np.zeros(num_batch), x_label="Iteration", y_label="Gradient", names=range(runs))
