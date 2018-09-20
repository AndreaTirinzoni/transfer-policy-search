import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

# def plot_value_function(V, title="Value Function"):
    # """
    # Plots the value function as a surface plot.
    # """
    # min_x = min(k[0] for k in V.keys())
    # max_x = max(k[0] for k in V.keys())
    # min_y = min(k[1] for k in V.keys())
    # max_y = max(k[1] for k in V.keys())
#
    # x_range = np.arange(min_x, max_x + 1)
    # y_range = np.arange(min_y, max_y + 1)
    # X, Y = np.meshgrid(x_range, y_range)
#
#    Find value for all (x, y) coordinates
    # Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    # Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))
#
    # def plot_surface(X, Y, Z, title):
        # fig = plt.figure(figsize=(20, 10))
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               # cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        # ax.set_xlabel('Player Sum')
        # ax.set_ylabel('Dealer Showing')
        # ax.set_zlabel('Value')
        # ax.set_title(title)
        # ax.view_init(ax.elev, -120)
        # fig.colorbar(surf)
        # plt.show()
#
    # plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    # plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


def plot_episode_stats(stats, stats_opt, num_episodes, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    # fig1 = plt.figure(figsize=(10,5))
    # plt.plot(range(num_episodes), stats.episode_lengths, linewidth=3)
    # plt.plot(range(num_episodes), stats_opt.episode_lengths, 'red', linewidth=1)
    # plt.xlabel("Episode")
    # plt.ylabel("Episode Length")
    # plt.title("Episode Length over Time")
    # if noshow:
        # plt.close(fig1)
    # else:
        # plt.show(fig1)

    # Plot the episode reward over time
    fig1 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_opt = pd.Series(stats_opt.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(range(num_episodes), rewards_smoothed, linewidth=3, label='REINFORCE')
    plt.plot(range(num_episodes), rewards_smoothed_opt, 'red', linewidth=0.5, label='Optimal policy')
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.legend()
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    return fig1

def plot_algorithm_comparison(stats_alg1, stats_alg2, stats_opt, num_episodes, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    # fig1 = plt.figure(figsize=(10,5))
    # plt.plot(range(num_episodes), stats.episode_lengths, linewidth=3)
    # plt.plot(range(num_episodes), stats_opt.episode_lengths, 'red', linewidth=1)
    # plt.xlabel("Episode")
    # plt.ylabel("Episode Length")
    # plt.title("Episode Length over Time")
    # if noshow:
        # plt.close(fig1)
    # else:
        # plt.show(fig1)

    # Plot the episode reward over time
    fig1 = plt.figure(figsize=(10,5))
    rewards_smoothed_alg1 = pd.Series(stats_alg1.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_opt = pd.Series(stats_opt.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_alg2 = pd.Series(stats_alg2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(range(num_episodes), rewards_smoothed_alg1, linewidth=3, label='REINFORCE')
    plt.plot(range(num_episodes), rewards_smoothed_opt, 'red', linewidth=0.5, label='Optimal policy')
    plt.plot(range(num_episodes), rewards_smoothed_alg2, 'green', linewidth=1.5, label='REINFORCE with baseline')
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.legend()
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    return fig1
