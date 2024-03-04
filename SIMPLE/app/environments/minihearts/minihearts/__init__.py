from gym.envs.registration import register

register(
    id='MiniHearts-v0',
    entry_point='hearts.envs:MiniHeartsEnv',
    max_episode_steps=1,
)