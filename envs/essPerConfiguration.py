import numpy as np
import importanceWeights as iw

def essPerTarget(variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, source_param, source_task):
    """
    The function computes eh ess for every combination of environment_parameter and policy_parameter
    :param variance_action: variance of the action distribution
    :param env_param_min: minimum value assumed by the environment parameter
    :param env_param_max: maximum value assumed by the environment parameter
    :param policy_param_min: minimum value assumed by the policy parameter
    :param policy_param_max: maximum value assumed by the policy parameter
    :param source_param: data structure to collect the parameters of the episode [policy_parameter, environment_parameter, environment_variance]
    :param source_task: data structure to collect informations about the episodes, every row contains all [state, action, reward, .....]
    :param episode_length: lenght of the episodes
    :return:

    A matrix containing ESS for every env_parameter - policy_parameter combination w.r.t the source task dataset
    """
    policy_param = np.linspace(policy_param_min, policy_param_max, 40)
    env_param = np.linspace(env_param_min, env_param_max, 160)
    ess = np.zeros((env_param.shape[0], policy_param.shape[0]))
    for i_policy_param in range(policy_param.shape[0]):
        for i_env_param in range(env_param.shape[0]):
            print(i_policy_param, i_env_param)
            weights_per_configuration = iw.computeImportanceWeightsSourceTarget(policy_param[i_policy_param], env_param[i_env_param], source_param, variance_action, source_task)
            ess[i_env_param, i_policy_param] = np.linalg.norm(weights_per_configuration, 1)**2 / np.linalg.norm(weights_per_configuration, 2)**2
    return ess

episode_length = 50
mean_initial_param = 0
variance_initial_param = 0
variance_action = 0.001
np.random.seed(2000)
num_episodes=1000
batch_size=40
discount_factor = 0.99
env_param_min = -2
env_param_max = 2
policy_param_min = -1
policy_param_max = 0

print("Loading files")
source_task = np.genfromtxt('source_task.csv', delimiter=',')
episodes_per_config = np.genfromtxt('episodes_per_config.csv', delimiter=',').astype(int)
source_param = np.genfromtxt('source_param.csv', delimiter=',')

print("Computing ESS")
ess = essPerTarget(variance_action, env_param_min, env_param_max, policy_param_min, policy_param_max, source_param, source_task)
np.savetxt("ess_source_tasks.csv", ess, delimiter=",")
