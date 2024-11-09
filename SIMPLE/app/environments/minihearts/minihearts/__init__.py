from gym.envs.registration import register

register(
    id='MiniHearts-v0',
    entry_point='minihearts.envs:MiniHeartsEnv',
)