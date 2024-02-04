#from Hearts import Hearts
import gym
import random


env = gym.make("hearts.envs:Hearts-v0")
#env.__init__(False, False)

observation, info = env.reset()
env.render()

steps = 0
done = False 
while not done:

    if steps == 47:
        print("last round")
    
    # make each agent select an action
    current_player = env.current_player_num
    legal_actions = env.legal_actions()

    # generate a random valid action
    action_index = random.randint(0, len(legal_actions) - 1)
    action_index = legal_actions[action_index]
    

    observation, reward, done, _, _ = env.step(observation['player_cards'][action_index])
    steps += 1