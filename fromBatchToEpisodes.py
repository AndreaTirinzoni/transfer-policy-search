import numpy as np

def batchToEpisodes(statistic_batch, episodesPerBatch, linspace_episodes):

    tot_episodes = np.sum(episodesPerBatch)
    statistic_episode = np.ones(tot_episodes)

    initial_index = 0

    for i in range(tot_episodes):
        episode_current_batch = episodesPerBatch[i]
        statistic_episode[initial_index:initial_index+episode_current_batch] = statistic_batch
        initial_index += episode_current_batch

    return statistic_episode[0::linspace_episodes]
