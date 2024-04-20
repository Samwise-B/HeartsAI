import ast
import numpy as np
import os

def test_stats():
    player_scores = [0, 0, 0, 0]
    player_win_count = [0, 0, 0, 0]
    player_loss_count = [0, 0, 0, 0]
    num_games = 0
    with open("SIMPLE/app/logs/log.txt") as fp:
        for i, line in enumerate(fp):
            if i > 6 and "games:" in line:
                num_games += 1
                _, score_obj = line.split("games: ")
                score_obj = ast.literal_eval(score_obj)
                scores = list(score_obj.values())
                for j in range(4):
                    # subtract accumulated scores for each player
                    scores[j] -= player_scores[j]
                    # track accumulated scores for each player
                    player_scores[j] += scores[j]
                
                winner = np.argmax(scores)
                loser = np.argmin(scores)
                player_win_count[winner] += 1
                player_loss_count[loser] += 1

    avg_scores = []
    for i in range(4):
        avg_scores.append(player_scores[i] / num_games)


    print(f"player wins: {str(player_win_count)}")
    print(f"player losses: {str(player_loss_count)}")
    print(f"player average score out of {num_games} games: {str(avg_scores)}")
    print(f"player total scores: {str(player_scores)}")

def run_tournament():
    env = "minihearts"
    zoo_dir = os.fsencode(f"./SIMPLE/app/zoo/{env}/")

    for p1 in os.listdir(zoo_dir):
        for p2 in os.listdir(zoo_dir):
            pass

def debug():
    reward = 0
    rewards = [0] * 4
    with open("SIMPLE/app/logs/log.txt") as fp:
        for i, line in enumerate(fp):
            if "Reward To Agent" in line:
                _, new_reward = line.split(":")
                new_reward = ast.literal_eval(new_reward[:-1])
                reward += new_reward
            elif "Rewards: " in line:
                _, new_rewards = line.split(":")
                new_rewards = ast.literal_eval(new_rewards[:-1])
                for i, r in enumerate(new_rewards):
                    rewards[i] += r
            elif "Agent plays as Player" in line:
                print(f"Agent is playing {line.split(' ')[-1][0]}")
            elif "Done: " in line:
                _, flag = line.split(":")
                flag = ast.literal_eval(flag[:-1])
                if flag == True:
                    print(f"accumulated agent rewards: {reward}")
                    print(f"accumulated rewards: {rewards}")
                    reward = 0
                    rewards = [0] * 4

            
def main():
    test_stats()
    #debug()

if __name__=="__main__":
    main()