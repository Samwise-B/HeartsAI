from gym.envs.registration import register

register(
    id='Hearts-v0',
    entry_point='hearts.envs:HeartsEnv',
    max_episode_steps=1,
)