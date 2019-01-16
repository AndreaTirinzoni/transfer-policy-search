from gym.envs.registration import register

register(
    id='cartpolec-v0',
    entry_point='envs.continuousCartpole:ContinuousCartPoleEnv'
)

register(
    id='LQG1D-v0',
    entry_point='envs.lqg1d:LQG1D'
)

register(
    id='testing-v0',
    entry_point='envs.test_env:TestEnv'
)

register(
    id='testenv-v0',
    entry_point='envs.test:TestEnv'
)

register(
    id='minigolf-v0',
    entry_point='envs.minigolf:MiniGolf'
)