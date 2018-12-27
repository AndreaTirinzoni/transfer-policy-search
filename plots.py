import pickle
from utils import plot
import numpy as np

with open('rewards.pkl', 'rb') as input:
    rewards = pickle.load(input)

with open('policy.pkl', 'rb') as input:
    policy = pickle.load(input)

with open('gradient.pkl', 'rb') as input:
    gradient = pickle.load(input)

with open('ess.pkl', 'rb') as input:
    ess = pickle.load(input)

with open('n_def.pkl', 'rb') as input:
    n_def = pickle.load(input)

estimators = ["MIS", "MIS-CV", "MIS-CV-BASELINE", "REINFORCE-BASELINE"]
runs = 20
num_batch = 200

x = range(num_batch)

from scipy.stats import t
alpha = t.interval(0.95, runs-1, loc=0, scale=1)[1] if runs > 1 else 1

# means = [np.mean(rewards[estimator], axis=0) for estimator in estimators]
# stds = [alpha * np.std(rewards[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]
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

plot.plot_curves([x for _ in estimators], [means[estimator][:, 0] for estimator in range(3)], [stds[estimator][:, 0] for estimator in range(3)], x_label="Iteration", y_label="Policy", names=estimators)
# means = [np.mean(rewards[estimator], axis=0) for estimator in estimators]
# stds = [alpha * np.std(rewards[estimator], axis=0) / np.sqrt(runs) for estimator in estimators]
#
# plot.plot_curves([x for _ in estimators], means, stds, x_label="Iteration", y_label="Return", names=estimators)
