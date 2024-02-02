from gym.envs.registration import register

register(
    id='Hearts-v0',
    entry_point='HeartsEnv',
    max_episode_steps=1,
)