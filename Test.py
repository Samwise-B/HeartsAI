from Hearts import Hearts
import gym


env = Hearts.HeartsEnv()
env.__init__(False, False)

observation = env.reset()

while True:
    env.render()

    # make each agent select an action
    #env.current_player_num

    # apply env.step

    break