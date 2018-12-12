from gym.envs.registration import register

register(
    id='cartpolec-v0',
    entry_point='envs.continuousCartpole:ContinuousCartPoleEnv'
)

register(
    id='LQG1D-v0',
    entry_point='envs.lqg1d:LQG1D'
)
