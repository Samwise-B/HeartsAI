import ast
import numpy as np

def main():
    player_scores = [0, 0, 0, 0]
    player_win_count = [0, 0, 0, 0]
    player_loss_count = [0, 0, 0, 0]
    num_games = 0
    with open("SIMPLE/app/logs/log.txt") as fp:
        for i, line in enumerate(fp):
            if i > 6:
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

if __name__=="__main__":
    main()